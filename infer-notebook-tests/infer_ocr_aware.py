# ═══════════════════════════════════════════════════════════════════════════
# OCR-AWARE ENSEMBLE INFERENCE
# Each adapter has its own USE_OCR flag — Run E was trained WITH ocr text in
# the prompt, Runs A/B/C were trained WITHOUT. Scoring an adapter under the
# wrong prompt distribution costs 5+ points, so we flag per-adapter.
#
# To submit Run E alone first (recommended): comment out A/B/C entries below.
# To submit a 4-way ensemble: leave them all in.
#
# Paste this whole script as ONE Colab cell (or run as `!python infer_ocr_aware.py`).
# ═══════════════════════════════════════════════════════════════════════════
import os, json, gc, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ════════════ EDIT ME ════════════════════════════════════════════════════
DATA_DIR  = Path("/content/data")
OCR_JSON  = Path("/content/drive/MyDrive/scienceqa_run_final_overnight/ocr_text.json")
OUT_CSV   = Path("/content/submission.csv")

# Per-adapter config: (path, img_size, use_ocr, weight)
# weight is for weighted ensemble averaging — start equal; bump Run E if it
# turns out to dominate cleanly.
ADAPTERS = [
    # (Path("/content/drive/MyDrive/scienceqa/outputs_train/adapter"),                       384, False, 1.0),  # Run A
    # (Path("/content/drive/MyDrive/scienceqa_run_final_overnight/seed7_res512"),            512, False, 1.0),  # Run B
    # (Path("/content/drive/MyDrive/scienceqa_run_final_overnight/seed21_res512_CoT"),       512, False, 1.0),  # Run C
    (Path("/content/drive/MyDrive/scienceqa_run_final_overnight/seed99_res512_OCR_8ep"),   512, True,  1.0),  # Run E
]

TTA_PERMUTATIONS  = 4         # bump to 8 for the final submission
EVAL_BATCH        = 16        # bump higher if VRAM allows
MAX_CONTEXT_CHARS = 1200
OCR_MAX_CHARS     = 500
MODEL_ID          = "HuggingFaceTB/SmolVLM-500M-Instruct"
# ═════════════════════════════════════════════════════════════════════════

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except ImportError: pass

# Sanity
for a in ADAPTERS:
    assert a[0].exists(), f"missing: {a[0]}"
assert OCR_JSON.exists(), f"OCR JSON missing: {OCR_JSON}"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"adapters: {len(ADAPTERS)}  TTA={TTA_PERMUTATIONS}  device={device}")

# ── Load test + OCR ──────────────────────────────────────────────────────
test_df = pd.read_csv(DATA_DIR / "test.csv")
test_df["choices"] = test_df["choices"].apply(json.loads)
OCR_TEXT = json.loads(OCR_JSON.read_text())
n_ocr = sum(1 for v in OCR_TEXT.values() if v.strip())
print(f"test rows: {len(test_df)}  ocr cache: {n_ocr}/{len(OCR_TEXT)} have text")

CHOICE_LETTERS = "ABCDEFGHIJ"

def _truncate(s, n):
    s = str(s).strip()
    return s if len(s) <= n else s[:n-1] + "…"

def build_user_text(row, choices, *, use_ocr):
    """Must mirror EXACTLY the training-time prompt format for each adapter."""
    parts = []
    lec = row.get("lecture")
    if isinstance(lec, str) and lec.strip(): parts.append(_truncate(lec, MAX_CONTEXT_CHARS))
    hint = row.get("hint")
    if isinstance(hint, str) and hint.strip(): parts.append(_truncate(hint, MAX_CONTEXT_CHARS//2))
    if use_ocr:
        ocr = (OCR_TEXT.get(row["id"], "") or "").strip()
        if ocr:
            parts.append(f"Text extracted from image (via OCR): {ocr[:OCR_MAX_CHARS]}")
    ctx = "\n".join(parts)
    cl  = "\n".join(f"{CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    text = ""
    if ctx: text += f"Context:\n{ctx}\n\n"
    text += f"Question: {row['question']}\n\nChoices:\n{cl}\n\nAnswer with a single letter only."
    return text

def build_messages(row, choices, *, use_ocr):
    return [{"role":"user","content":[
        {"type":"image"},
        {"type":"text","text":build_user_text(row, choices, use_ocr=use_ocr)},
    ]}]

def load_image(rel, img_size):
    im = Image.open(DATA_DIR / rel).convert("RGB")
    w, h = im.size; s = img_size / max(w, h)
    if s < 1.0: im = im.resize((max(1,int(w*s)), max(1,int(h*s))), Image.BICUBIC)
    return im

# Pre-build row dicts
test_rows = [{
    "id": r["id"], "image_path": r["image_path"], "question": r["question"],
    "choices": r["choices"], "lecture": r.get("lecture"), "hint": r.get("hint"),
} for _, r in test_df.iterrows()]

# ── Load base model ONCE (saves a few minutes per adapter) ───────────────
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

print("loading base model + processor (once)...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.padding_side = "left"

base = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
)
base.eval(); base.config.use_cache = True

def letter_id(L):
    ids = processor.tokenizer(L, add_special_tokens=False).input_ids
    return ids[-1] if len(ids) >= 1 else processor.tokenizer(" "+L, add_special_tokens=False).input_ids[-1]
LIDS = {L: letter_id(L) for L in CHOICE_LETTERS}

# ── Score one adapter end-to-end ─────────────────────────────────────────
def score_with_adapter(adapter_dir, img_size, use_ocr, tta_perms):
    print(f"\n=== {adapter_dir.name}  img={img_size}  ocr={use_ocr}  TTA={tta_perms} ===")
    t0 = time.time()
    # Load adapter onto base
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval(); model.config.use_cache = True

    images = [load_image(r["image_path"], img_size) for r in test_rows]
    summed = [np.zeros(len(r["choices"]), dtype=np.float64) for r in test_rows]
    rng = np.random.RandomState(0)

    @torch.inference_mode()
    def score_pass(perms):
        for start in range(0, len(test_rows), EVAL_BATCH):
            sl = slice(start, min(start+EVAL_BATCH, len(test_rows)))
            rows_b = test_rows[sl]; imgs_b = images[sl]; perms_b = perms[sl]
            shuffled = [[r["choices"][i] for i in p] for r, p in zip(rows_b, perms_b)]
            prompts = [processor.apply_chat_template(
                build_messages(r, ch, use_ocr=use_ocr), add_generation_prompt=True
            ) for r, ch in zip(rows_b, shuffled)]
            inp = processor(text=prompts, images=[[im] for im in imgs_b],
                            return_tensors="pt", padding=True)
            inp = {k:(v.to(model.device) if torch.is_tensor(v) else v) for k,v in inp.items()}
            out = model(**inp); last = out.logits[:, -1, :]
            for i, p in enumerate(perms_b):
                ids = [LIDS[CHOICE_LETTERS[k]] for k in range(len(p))]
                row_logits = last[i, ids].float().cpu().numpy()
                gi = start + i
                for j, orig_idx in enumerate(p):
                    summed[gi][orig_idx] += float(row_logits[j])

    for k in range(tta_perms):
        perms = [(np.arange(len(r["choices"])) if k == 0 else rng.permutation(len(r["choices"])))
                 for r in test_rows]
        score_pass(perms)
        print(f"  pass {k+1}/{tta_perms}  elapsed={(time.time()-t0)/60:.1f}m")

    # average over passes
    out = [s / tta_perms for s in summed]
    # Unload adapter to free memory and prep for next
    model = model.unload() if hasattr(model, "unload") else model
    del model; gc.collect(); torch.cuda.empty_cache()
    return out

# ── Score all adapters, weighted ensemble ────────────────────────────────
final = [None] * len(test_rows)
total_w = sum(a[3] for a in ADAPTERS)
for adapter_dir, img_size, use_ocr, w in ADAPTERS:
    logits_list = score_with_adapter(adapter_dir, img_size, use_ocr, TTA_PERMUTATIONS)
    for i, lg in enumerate(logits_list):
        if final[i] is None: final[i] = np.zeros_like(lg)
        final[i] += (w / total_w) * lg

preds = [int(np.argmax(s)) for s in final]
print(f"\npreds done: {len(preds)}")

# ── Submission ───────────────────────────────────────────────────────────
sub = pd.DataFrame({"id":[r["id"] for r in test_rows], "answer": preds})
assert list(sub.columns) == ["id","answer"]
assert len(sub) == len(test_df)
assert set(sub["id"]) == set(test_df["id"])
assert sub["answer"].dtype.kind in ("i","u")
for r, p in zip(test_rows, preds):
    assert 0 <= p < len(r["choices"])
sub.to_csv(OUT_CSV, index=False)
print(f"\n✅ wrote {OUT_CSV}  ({len(sub)} rows)")
print("\nAnswer distribution:")
print(sub["answer"].value_counts().sort_index())
