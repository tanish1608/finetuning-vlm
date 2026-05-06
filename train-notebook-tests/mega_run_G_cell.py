# ═══════════════════════════════════════════════════════════════════════════
# RUN G — same recipe as Run E (the 0.90342 winner) but uses the PADDLEOCR
# cache instead of EasyOCR. This is a controlled experiment: if Run G beats E
# individually on the LB, PaddleOCR was the difference.
#
# Prereqs: Phase 1 cells already executed (cfg, ScienceQADataset, etc. in scope),
# AND paddleocr_preprocess_cell.py has been run (ocr_text_paddle.json on Drive).
# ═══════════════════════════════════════════════════════════════════════════
import os, gc, json, time, random, shutil, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

# Free leftover GPU memory
for v in ("model", "optimizer", "scheduler", "trainable_params", "base_model"):
    if v in dir(): exec(f"del {v}")
gc.collect(); torch.cuda.empty_cache()
print(f"GPU free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

# Drive
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except ImportError: pass

# ── Paths ────────────────────────────────────────────────────────────────
DRIVE_ROOT = Path("/content/drive/MyDrive/scienceqa_run_final_overnight")
OCR_JSON   = DRIVE_ROOT / "ocr_text_paddle.json"           # ← NEW (PaddleOCR)
RUN_G_DIR  = DRIVE_ROOT / "seed99_res512_PaddleOCR_8ep"    # ← NEW output dir
RUN_G_DIR.mkdir(parents=True, exist_ok=True)
assert OCR_JSON.exists(), f"missing {OCR_JSON} — run paddleocr_preprocess_cell.py first"

OCR_TEXT = json.loads(OCR_JSON.read_text())
n_with = sum(1 for v in OCR_TEXT.values() if v.strip())
print(f"PaddleOCR cache: {n_with:,}/{len(OCR_TEXT):,} images have text")

# ── Settings (mirror Run E exactly except OCR source) ────────────────────
SEED_G = 99               # SAME as E so we isolate "PaddleOCR vs EasyOCR" as the only diff
random.seed(SEED_G); np.random.seed(SEED_G); torch.manual_seed(SEED_G); torch.cuda.manual_seed_all(SEED_G)

cfg.epochs       = 8
cfg.img_size     = 512
cfg.lr           = 2e-4
cfg.micro_bsz    = 4
cfg.grad_accum   = 4
cfg.adapter_dir  = Path("outputs_train/adapter_run_G")
cfg.adapter_dir.mkdir(parents=True, exist_ok=True)

OCR_MAX_CHARS = 500       # match Run E's training distribution

# ── Train + 80% val combined; 420-row monitor ───────────────────────────
val_shuffled = val_df.sample(frac=1, random_state=SEED_G).reset_index(drop=True)
N_MONITOR    = 420
val_monitor  = val_shuffled.iloc[:N_MONITOR].reset_index(drop=True)
val_train    = val_shuffled.iloc[N_MONITOR:].reset_index(drop=True)
combined_train = pd.concat([train_df, val_train], ignore_index=True).sample(frac=1, random_state=SEED_G).reset_index(drop=True)
print(f"Combined train: {len(combined_train):,} | Monitor: {len(val_monitor):,}")

# ── Prompt builder using the PADDLE OCR cache ────────────────────────────
def build_user_text_paddle(row, *, max_ctx=cfg.max_context_chars,
                            use_lecture=True, use_hint=True, choices=None):
    if choices is None: choices = row["choices"]
    parts = []
    if use_lecture:
        lec = row.get("lecture", None)
        if isinstance(lec, str) and lec.strip(): parts.append(_truncate(lec, max_ctx))
    if use_hint:
        hint = row.get("hint", None)
        if isinstance(hint, str) and hint.strip(): parts.append(_truncate(hint, max_ctx//2))
    ocr = (OCR_TEXT.get(row["id"], "") or "").strip()
    if ocr:
        parts.append(f"Text extracted from image (via OCR): {ocr[:OCR_MAX_CHARS]}")
    context = "\n".join(parts)
    cl = "\n".join(f"{CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    text = ""
    if context: text += f"Context:\n{context}\n\n"
    text += f"Question: {row['question']}\n\nChoices:\n{cl}\n\nAnswer with a single letter only."
    return text

def build_messages_paddle(row, *, include_answer, choices=None, answer_idx=None):
    user_text = build_user_text_paddle(
        row, max_ctx=cfg.max_context_chars,
        use_lecture=cfg.use_lecture, use_hint=cfg.use_hint, choices=choices,
    )
    msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":user_text}]}]
    if include_answer:
        idx = answer_idx if answer_idx is not None else int(row["answer"])
        msgs.append({"role":"assistant","content":[{"type":"text","text":CHOICE_LETTERS[idx]}]})
    return msgs

# ── Collator + scoring (Paddle-OCR aware) ────────────────────────────────
def make_train_batch_paddle(items):
    full_texts, prompt_texts, images = [], [], []
    for it in items:
        mf = build_messages_paddle(it["row"], include_answer=True,
                                    choices=it["choices"], answer_idx=it["answer"])
        mp = build_messages_paddle(it["row"], include_answer=False, choices=it["choices"])
        full_texts.append(processor.apply_chat_template(mf, add_generation_prompt=False))
        prompt_texts.append(processor.apply_chat_template(mp, add_generation_prompt=True))
        images.append([it["image"]])
    processor.tokenizer.padding_side = "right"
    enc = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    plens = []
    for p, im in zip(prompt_texts, images):
        pe = processor(text=p, images=im, return_tensors="pt")
        plens.append(int(pe["input_ids"].shape[1]))
    labels = enc["input_ids"].clone()
    for i, pl in enumerate(plens): labels[i, :pl] = -100
    labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels
    processor.tokenizer.padding_side = "left"
    return enc

@torch.inference_mode()
def score_batch_paddle(model_, items):
    prompts, images, ncs = [], [], []
    for it in items:
        msgs = build_messages_paddle(it["row"], include_answer=False, choices=it["choices"])
        prompts.append(processor.apply_chat_template(msgs, add_generation_prompt=True))
        images.append([it["image"]]); ncs.append(len(it["choices"]))
    processor.tokenizer.padding_side = "left"
    inp = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inp = {k:(v.to(model_.device) if torch.is_tensor(v) else v) for k,v in inp.items()}
    out = model_(**inp); last = out.logits[:, -1, :]
    preds = []
    for i, n in enumerate(ncs):
        ids = [LETTER_IDS[CHOICE_LETTERS[k]] for k in range(n)]
        preds.append(int(last[i, ids].argmax().item()))
    return np.array(preds)

def evaluate_paddle(model_, dataset, max_samples=None, bs=16):
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    correct = total = 0
    model_.eval()
    for s in range(0, n, bs):
        items = [dataset[s+j] for j in range(min(bs, n-s))]
        gts = np.array([it["answer"] for it in items])
        correct += int((score_batch_paddle(model_, items) == gts).sum()); total += len(items)
    return {"accuracy": correct/max(1,total), "n": total}

# ── Datasets / loader ────────────────────────────────────────────────────
train_ds = ScienceQADataset(combined_train, cfg.data_dir, img_size=cfg.img_size,
                             is_train=True, shuffle_choices=True)
val_ds   = ScienceQADataset(val_monitor,   cfg.data_dir, img_size=cfg.img_size, is_train=False)
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=cfg.micro_bsz, shuffle=True,
    collate_fn=make_train_batch_paddle, num_workers=0, pin_memory=True, drop_last=True,
)
print(f"steps/epoch={len(train_loader)}  effective_batch={cfg.micro_bsz*cfg.grad_accum}")

# ── Fresh base + fresh LoRA (same topology as Run E) ────────────────────
from transformers import AutoModelForImageTextToText, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
base_model = AutoModelForImageTextToText.from_pretrained(
    cfg.model_id, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
)
for p in base_model.parameters(): p.requires_grad = False
base_model.config.use_cache = False
lora_cfg_G = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(base_model, lora_cfg_G)
model.gradient_checkpointing_enable()
if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}"); assert trainable <= 5_000_000

# ── Optim + cosine ───────────────────────────────────────────────────────
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
total_update_steps = (len(train_loader) // cfg.grad_accum) * cfg.epochs
warmup_steps = max(1, int(total_update_steps * cfg.warmup_ratio))
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)
print(f"total_updates={total_update_steps}  warmup={warmup_steps}  lr_peak={cfg.lr}")

# ── Save-best mirror ─────────────────────────────────────────────────────
def _save_best_g(acc, step):
    global best_acc, best_step
    if acc > best_acc:
        best_acc, best_step = acc, step
        model.save_pretrained(cfg.adapter_dir)
        processor.save_pretrained(cfg.adapter_dir)
        for f in Path(cfg.adapter_dir).glob("*"):
            if f.is_file(): shutil.copy2(f, RUN_G_DIR / f.name)
        print(f"  ★ new best ({acc*100:.2f}%) — mirrored to Drive: {RUN_G_DIR}")

# ── Train ────────────────────────────────────────────────────────────────
history = {"step":[], "loss":[], "eval_step":[], "eval_acc":[]}
best_acc, best_step = -1.0, -1
update_step = micro_step = 0
running_loss = 0.0; loss_count = 0
t0 = time.time()

for epoch in range(cfg.epochs):
    print(f"\n=== Run G (PaddleOCR)  Epoch {epoch+1}/{cfg.epochs} ===")
    model.train(); optimizer.zero_grad(set_to_none=True)
    for batch in train_loader:
        batch = {k:(v.to(model.device) if torch.is_tensor(v) else v) for k,v in batch.items()}
        out = model(**batch)
        loss = out.loss / cfg.grad_accum
        loss.backward()
        running_loss += loss.item() * cfg.grad_accum; loss_count += 1
        micro_step += 1
        if micro_step % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            update_step += 1
            if update_step % cfg.log_every == 0:
                avg = running_loss / max(1, loss_count); running_loss = 0.0; loss_count = 0
                history["step"].append(update_step); history["loss"].append(avg)
                print(f"  step {update_step:>5d}/{total_update_steps}  loss={avg:.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}  elapsed={(time.time()-t0)/60:.1f}m")
            if update_step % cfg.eval_every == 0:
                ev = evaluate_paddle(model, val_ds)
                history["eval_step"].append(update_step); history["eval_acc"].append(ev["accuracy"])
                print(f"  ↪ monitor acc (n={ev['n']}): {ev['accuracy']*100:.2f}%")
                _save_best_g(ev["accuracy"], update_step); model.train()
    ev = evaluate_paddle(model, val_ds)
    history["eval_step"].append(update_step); history["eval_acc"].append(ev["accuracy"])
    print(f"--- end epoch {epoch+1} monitor acc (n={ev['n']}): {ev['accuracy']*100:.2f}%")
    _save_best_g(ev["accuracy"], update_step)

print(f"\n✅ Run G done in {(time.time()-t0)/60:.1f} min — best monitor = {best_acc*100:.2f}%")

# ── Save meta ────────────────────────────────────────────────────────────
(RUN_G_DIR / "meta.json").write_text(json.dumps({
    "run":"G_PaddleOCR_8ep",
    "seed": SEED_G, "img_size": cfg.img_size, "epochs": cfg.epochs, "lr": cfg.lr,
    "best_monitor_acc": float(best_acc), "best_step": int(best_step),
    "trainable_params": int(trainable),
    "elapsed_min": round((time.time()-t0)/60, 1),
    "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "training_data": {"n_train_orig": int(len(train_df)),
                       "n_val_added": int(len(val_train)),
                       "monitor_n": int(len(val_monitor))},
    "ocr_source": "PaddleOCR", "ocr_max_chars": OCR_MAX_CHARS,
}, indent=2))
(RUN_G_DIR / "history.json").write_text(json.dumps(history))

# ── Disconnect ───────────────────────────────────────────────────────────
print("Sleeping 60s for Drive flush…")
time.sleep(60)
try:
    from google.colab import runtime; runtime.unassign()
except Exception as e:
    print(e); os._exit(0)
