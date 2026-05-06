"""
overnight_train.py — Train two SmolVLM LoRA adapters back-to-back with
different seeds & configs, save both to Google Drive, then disconnect the
Colab runtime so you stop burning credits.

USAGE in a Colab cell:
    !python /content/overnight_train.py
    # or, to keep the cell's namespace:
    %run /content/overnight_train.py

EDIT the section marked "USER SETTINGS" before running.
"""

import os, json, time, math, random, shutil, gc, datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# ════════════════════════════════════════════════════════════════════════════
# USER SETTINGS — edit these to match your environment
# ════════════════════════════════════════════════════════════════════════════
DATA_DIR    = Path("/content/data")                            # train.csv, val.csv, test.csv, images/
DRIVE_ROOT  = Path("/content/drive/MyDrive/scienceqa_run")     # where adapters land
LOCAL_OUT   = Path("/content/outputs_train")                   # local working dir
MODEL_ID    = "HuggingFaceTB/SmolVLM-500M-Instruct"
DISCONNECT_AT_END = True                                        # set False to keep runtime alive
# ════════════════════════════════════════════════════════════════════════════

CHOICE_LETTERS = "ABCDEFGHIJ"

@dataclass
class RunCfg:
    name: str
    seed: int
    img_size: int
    epochs: int
    lr: float
    lora_r: int
    lora_alpha: int
    lora_target_modules: tuple
    micro_bsz: int = 4
    grad_accum: int = 4
    eval_bsz:  int = 8
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    max_context_chars: int = 1200
    use_lecture: bool = True
    use_hint:    bool = True
    shuffle_choices_train: bool = True
    log_every:  int = 25
    eval_every: int = 250
    eval_max_samples: int = 600
    end_of_epoch_eval_max: int = 1500

# Two diverse runs
RUNS = [
    RunCfg(
        name="seed42_res384_attnLoRA16",
        seed=42, img_size=384, epochs=3, lr=2e-4,
        lora_r=16, lora_alpha=32,
        lora_target_modules=("q_proj","k_proj","v_proj","o_proj"),
    ),
    RunCfg(
        name="seed7_res512_attnMLPLoRA",
        seed=7, img_size=512, epochs=3, lr=2e-4,
        lora_r=8, lora_alpha=16,
        # attention r=8 + MLP r=4-equivalent (peft uses single r; we keep r=8 across modules)
        lora_target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"),
    ),
]

# ── Mount Drive (no-op if already mounted) ─────────────────────────────────
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    print("✅ Drive mounted")
except ImportError:
    print("ℹ️  Not in Colab — Drive mount skipped")

LOCAL_OUT.mkdir(parents=True, exist_ok=True)
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

# ── Load CSVs once ─────────────────────────────────────────────────────────
print("Loading CSVs...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df   = pd.read_csv(DATA_DIR / "val.csv")
for df in [train_df, val_df]:
    df["choices"] = df["choices"].apply(json.loads)
print(f"  train={len(train_df)}  val={len(val_df)}")

# ── Prompt construction ────────────────────────────────────────────────────
def _truncate(s: str, n: int) -> str:
    s = str(s).strip()
    return s if len(s) <= n else s[: n-1] + "…"

def build_user_text(row, choices, *, max_ctx, use_lecture, use_hint):
    parts = []
    if use_lecture:
        lec = row.get("lecture", None)
        if isinstance(lec, str) and lec.strip(): parts.append(_truncate(lec, max_ctx))
    if use_hint:
        hint = row.get("hint", None)
        if isinstance(hint, str) and hint.strip(): parts.append(_truncate(hint, max_ctx//2))
    context = "\n".join(parts)
    choice_lines = "\n".join(f"{CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    text = ""
    if context: text += f"Context:\n{context}\n\n"
    text += f"Question: {row['question']}\n\nChoices:\n{choice_lines}\n\nAnswer with a single letter only."
    return text

def build_messages(row, choices, *, include_answer, answer_idx, max_ctx, use_lecture, use_hint):
    msgs = [{"role":"user","content":[
        {"type":"image"},
        {"type":"text","text":build_user_text(row, choices, max_ctx=max_ctx,
                                              use_lecture=use_lecture, use_hint=use_hint)},
    ]}]
    if include_answer:
        msgs.append({"role":"assistant","content":[
            {"type":"text","text":CHOICE_LETTERS[answer_idx]},
        ]})
    return msgs

class ScienceQADataset(Dataset):
    def __init__(self, df, *, img_size, is_train, shuffle_choices):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.is_train = is_train
        self.shuffle_choices = shuffle_choices
    def __len__(self): return len(self.df)
    def _img(self, p):
        im = Image.open(DATA_DIR / p).convert("RGB")
        w,h = im.size; s = self.img_size / max(w,h)
        if s < 1.0: im = im.resize((max(1,int(w*s)), max(1,int(h*s))), Image.BICUBIC)
        return im
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = self._img(r["image_path"])
        ch = list(r["choices"])
        ans = int(r["answer"]) if "answer" in r and not pd.isna(r["answer"]) else -1
        if self.is_train and self.shuffle_choices and ans >= 0:
            perm = np.random.permutation(len(ch))
            ch = [ch[k] for k in perm]
            ans = int(np.where(perm == ans)[0][0])
        return {"id": r["id"], "image": img, "row": r, "choices": ch, "answer": ans}

# ── One full run ──────────────────────────────────────────────────────────
def run_one(rcfg: RunCfg):
    print(f"\n{'='*72}\n🚀 RUN: {rcfg.name}\n{'='*72}")
    print(json.dumps({k:str(v) for k,v in asdict(rcfg).items()}, indent=2))

    # set seeds
    random.seed(rcfg.seed); np.random.seed(rcfg.seed)
    torch.manual_seed(rcfg.seed); torch.cuda.manual_seed_all(rcfg.seed)

    from transformers import AutoProcessor, AutoModelForImageTextToText, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    for p in base.parameters(): p.requires_grad = False
    base.config.use_cache = False

    lora_cfg = LoraConfig(
        r=rcfg.lora_r, lora_alpha=rcfg.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=list(rcfg.lora_target_modules),
    )
    model = get_peft_model(base, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    if trainable > 5_000_000:
        raise RuntimeError(f"Over 5M trainable params: {trainable}")
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()

    # letter token ids
    def letter_id(L):
        ids = processor.tokenizer(L, add_special_tokens=False).input_ids
        return ids[-1] if len(ids) >= 1 else processor.tokenizer(" "+L, add_special_tokens=False).input_ids[-1]
    LIDS = {L: letter_id(L) for L in CHOICE_LETTERS}

    train_ds = ScienceQADataset(train_df, img_size=rcfg.img_size, is_train=True,  shuffle_choices=rcfg.shuffle_choices_train)
    val_ds   = ScienceQADataset(val_df,   img_size=rcfg.img_size, is_train=False, shuffle_choices=False)

    @torch.inference_mode()
    def score_batch(items):
        prompts, images, ncs = [], [], []
        for it in items:
            msgs = build_messages(it["row"], it["choices"], include_answer=False, answer_idx=0,
                                  max_ctx=rcfg.max_context_chars, use_lecture=rcfg.use_lecture, use_hint=rcfg.use_hint)
            prompts.append(processor.apply_chat_template(msgs, add_generation_prompt=True))
            images.append([it["image"]]); ncs.append(len(it["choices"]))
        processor.tokenizer.padding_side = "left"
        inp = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inp = {k:(v.to(model.device) if torch.is_tensor(v) else v) for k,v in inp.items()}
        out = model(**inp); last = out.logits[:, -1, :]
        preds = []
        for i, n in enumerate(ncs):
            ids = [LIDS[CHOICE_LETTERS[k]] for k in range(n)]
            preds.append(int(last[i, ids].argmax().item()))
        return np.array(preds)

    def evaluate(max_samples=None):
        n = len(val_ds) if max_samples is None else min(max_samples, len(val_ds))
        bs = rcfg.eval_bsz; correct = total = 0
        model.eval()
        for s in range(0, n, bs):
            items = [val_ds[s+j] for j in range(min(bs, n-s))]
            gts = np.array([it["answer"] for it in items])
            correct += int((score_batch(items) == gts).sum()); total += len(items)
        return correct/max(1,total), total

    def collate(items):
        full_t, prompt_t, images = [], [], []
        for it in items:
            mf = build_messages(it["row"], it["choices"], include_answer=True, answer_idx=it["answer"],
                                max_ctx=rcfg.max_context_chars, use_lecture=rcfg.use_lecture, use_hint=rcfg.use_hint)
            mp = build_messages(it["row"], it["choices"], include_answer=False, answer_idx=0,
                                max_ctx=rcfg.max_context_chars, use_lecture=rcfg.use_lecture, use_hint=rcfg.use_hint)
            full_t.append(processor.apply_chat_template(mf, add_generation_prompt=False))
            prompt_t.append(processor.apply_chat_template(mp, add_generation_prompt=True))
            images.append([it["image"]])
        processor.tokenizer.padding_side = "right"
        enc = processor(text=full_t, images=images, return_tensors="pt", padding=True)
        plens = []
        for p, im in zip(prompt_t, images):
            pe = processor(text=p, images=im, return_tensors="pt")
            plens.append(int(pe["input_ids"].shape[1]))
        labels = enc["input_ids"].clone()
        for i, pl in enumerate(plens): labels[i, :pl] = -100
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        processor.tokenizer.padding_side = "left"
        return enc

    loader = DataLoader(train_ds, batch_size=rcfg.micro_bsz, shuffle=True,
                        collate_fn=collate, num_workers=0, pin_memory=True, drop_last=True)
    print(f"steps/epoch={len(loader)}  effective_batch={rcfg.micro_bsz*rcfg.grad_accum}")

    trainable_p = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable_p, lr=rcfg.lr, weight_decay=rcfg.weight_decay)
    total_updates = (len(loader) // rcfg.grad_accum) * rcfg.epochs
    warmup = max(1, int(total_updates * rcfg.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(optim, warmup, total_updates)
    print(f"total_updates={total_updates}  warmup={warmup}")

    adapter_local = LOCAL_OUT / rcfg.name
    adapter_local.mkdir(parents=True, exist_ok=True)
    adapter_drive = DRIVE_ROOT / rcfg.name
    adapter_drive.mkdir(parents=True, exist_ok=True)

    best_acc, best_step = -1.0, -1
    history = {"step":[], "loss":[], "eval_step":[], "eval_acc":[]}
    update_step = micro_step = 0
    run_loss = 0.0; run_n = 0
    t0 = time.time()

    def save_best(acc, step):
        nonlocal best_acc, best_step
        if acc > best_acc:
            best_acc, best_step = acc, step
            model.save_pretrained(adapter_local)
            processor.save_pretrained(adapter_local)
            # mirror to Drive immediately so a crash can't lose the best
            for f in adapter_local.glob("*"):
                if f.is_file():
                    shutil.copy2(f, adapter_drive / f.name)
            print(f"  ★ new best ({acc*100:.2f}%) saved to Drive: {adapter_drive}")

    for epoch in range(rcfg.epochs):
        print(f"\n--- {rcfg.name}  Epoch {epoch+1}/{rcfg.epochs} ---")
        model.train(); optim.zero_grad(set_to_none=True)
        for batch in loader:
            batch = {k:(v.to(model.device) if torch.is_tensor(v) else v) for k,v in batch.items()}
            out = model(**batch)
            loss = out.loss / rcfg.grad_accum
            loss.backward()
            run_loss += loss.item() * rcfg.grad_accum; run_n += 1
            micro_step += 1
            if micro_step % rcfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_p, rcfg.max_grad_norm)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
                update_step += 1
                if update_step % rcfg.log_every == 0:
                    avg = run_loss / max(1, run_n); run_loss = 0; run_n = 0
                    history["step"].append(update_step); history["loss"].append(avg)
                    print(f"  step {update_step:>5d}/{total_updates}  loss={avg:.4f}  "
                          f"lr={sched.get_last_lr()[0]:.2e}  elapsed={(time.time()-t0)/60:.1f}m")
                if update_step % rcfg.eval_every == 0:
                    acc, n = evaluate(max_samples=rcfg.eval_max_samples)
                    history["eval_step"].append(update_step); history["eval_acc"].append(acc)
                    print(f"  ↪ val acc (n={n}): {acc*100:.2f}%")
                    save_best(acc, update_step)
                    model.train()
        # end-of-epoch full(er) eval
        acc, n = evaluate(max_samples=min(rcfg.end_of_epoch_eval_max, len(val_ds)))
        history["eval_step"].append(update_step); history["eval_acc"].append(acc)
        print(f"--- end of epoch {epoch+1} val acc (n={n}): {acc*100:.2f}%")
        save_best(acc, update_step)

    # write training history + meta to Drive
    with open(adapter_drive / "history.json", "w") as f: json.dump(history, f)
    meta = {
        "run_name": rcfg.name, "best_val_acc": best_acc, "best_step": best_step,
        "trainable_params": int(trainable),
        "elapsed_min": round((time.time()-t0)/60, 1),
        "config": {k:str(v) for k,v in asdict(rcfg).items()},
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(adapter_drive / "meta.json", "w") as f: json.dump(meta, f, indent=2)
    print(f"\n✅ {rcfg.name} done. best={best_acc*100:.2f}% in {(time.time()-t0)/60:.1f} min")
    print(f"   adapter -> {adapter_drive}")

    # free GPU memory between runs
    del model, base, processor, optim, sched, loader, trainable_p
    gc.collect(); torch.cuda.empty_cache()
    return meta

# ════════════════════════════════════════════════════════════════════════════
# Run all
# ════════════════════════════════════════════════════════════════════════════
all_meta = []
overall_t0 = time.time()
for rcfg in RUNS:
    try:
        all_meta.append(run_one(rcfg))
    except Exception as e:
        print(f"\n💥 RUN {rcfg.name} CRASHED: {e}")
        import traceback; traceback.print_exc()
        # keep going to next run

# Summary file
summary = {
    "runs": all_meta,
    "total_minutes": round((time.time()-overall_t0)/60, 1),
    "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
}
with open(DRIVE_ROOT / "OVERNIGHT_SUMMARY.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*72)
print("OVERNIGHT SUMMARY")
print("="*72)
print(json.dumps(summary, indent=2))

# Disconnect to stop credit burn
if DISCONNECT_AT_END:
    print("\nSleeping 60s to flush Drive…")
    time.sleep(60)
    print("Disconnecting runtime. 💤")
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception as e:
        print("runtime.unassign failed:", e)
        os._exit(0)
