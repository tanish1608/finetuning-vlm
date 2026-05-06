# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — Chain-of-thought continuation from Run B's adapter, on train + val
# Paste as ONE cell in your train.ipynb session AFTER all the cells from
# Phase 1 (so cfg, ScienceQADataset, build_user_text, train_collate, evaluate,
# processor, train_df, val_df are still in scope).
# ═══════════════════════════════════════════════════════════════════════════
import os, gc, json, time, random, shutil, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

# ── 0) Free GPU memory from any leftover model ───────────────────────────
for v in ("model", "optimizer", "scheduler", "trainable_params", "base_model"):
    if v in dir(): exec(f"del {v}")
gc.collect(); torch.cuda.empty_cache()
print(f"GPU free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

# ── 1) Mount Drive ───────────────────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except ImportError: pass

# ── 2) Settings ──────────────────────────────────────────────────────────
SEED_C = 21
random.seed(SEED_C); np.random.seed(SEED_C); torch.manual_seed(SEED_C); torch.cuda.manual_seed_all(SEED_C)

# ⚠️ EDIT to match your actual Drive paths
DRIVE_ROOT  = Path("/content/drive/MyDrive/scienceqa_run_final_overnight")
RUN_B_DIR   = DRIVE_ROOT / "seed7_res512"             # adapter to continue from
RUN_C_DIR   = DRIVE_ROOT / "seed21_res512_CoT"        # destination for Run C
RUN_C_DIR.mkdir(parents=True, exist_ok=True)
assert RUN_B_DIR.exists(), f"Run B adapter not found at {RUN_B_DIR}"

cfg.epochs       = 3
cfg.img_size     = 512
cfg.lr           = 5e-5         # low — we're continuing a converged adapter
cfg.adapter_dir  = Path("outputs_train/adapter_run_C")
cfg.adapter_dir.mkdir(parents=True, exist_ok=True)

# ── 3) Build combined train + val[80%] dataset ───────────────────────────
val_shuffled = val_df.sample(frac=1, random_state=SEED_C).reset_index(drop=True)
N_MONITOR    = 420
val_monitor  = val_shuffled.iloc[:N_MONITOR].reset_index(drop=True)
val_train    = val_shuffled.iloc[N_MONITOR:].reset_index(drop=True)
combined_train = pd.concat([train_df, val_train], ignore_index=True)
print(f"Combined train: {len(combined_train):,}  (train={len(train_df)} + val_train={len(val_train)})")
print(f"Monitor val   : {len(val_monitor):,}")

# ── 4) CoT-aware message builder + collator (override the originals) ─────
def build_messages_cot(row, choices, *, include_answer, answer_idx,
                       max_ctx, use_lecture, use_hint):
    user_text = build_user_text(row, choices, max_ctx=max_ctx,
                                 use_lecture=use_lecture, use_hint=use_hint)
    msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":user_text}]}]
    if include_answer:
        letter = CHOICE_LETTERS[answer_idx]
        sol = row.get("solution", None)
        if isinstance(sol, str) and sol.strip():
            sol_short = sol.strip()[:400]
            assistant_text = f"{letter}. {sol_short}"
        else:
            assistant_text = letter
        msgs.append({"role":"assistant","content":[{"type":"text","text":assistant_text}]})
    return msgs

def make_train_batch_cot(items):
    """Like make_train_batch, but uses CoT assistant turn (letter + solution)."""
    images, full_texts, prompt_texts = [], [], []
    for it in items:
        msgs_full = build_messages_cot(
            it["row"], it["choices"], include_answer=True, answer_idx=it["answer"],
            max_ctx=cfg.max_context_chars, use_lecture=cfg.use_lecture, use_hint=cfg.use_hint,
        )
        msgs_prompt = build_messages_cot(
            it["row"], it["choices"], include_answer=False, answer_idx=0,
            max_ctx=cfg.max_context_chars, use_lecture=cfg.use_lecture, use_hint=cfg.use_hint,
        )
        full_texts.append(processor.apply_chat_template(msgs_full, add_generation_prompt=False))
        prompt_texts.append(processor.apply_chat_template(msgs_prompt, add_generation_prompt=True))
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

train_ds = ScienceQADataset(combined_train, cfg.data_dir, img_size=cfg.img_size,
                            is_train=True, shuffle_choices=cfg.shuffle_choices_train)
val_ds   = ScienceQADataset(val_monitor,   cfg.data_dir, img_size=cfg.img_size, is_train=False)
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=cfg.micro_bsz, shuffle=True,
    collate_fn=make_train_batch_cot, num_workers=0, pin_memory=True, drop_last=True,
)
print(f"steps/epoch={len(train_loader)}  effective_batch={cfg.micro_bsz*cfg.grad_accum}")

# ── 5) Reload base + Run B adapter (trainable) ───────────────────────────
from transformers import AutoModelForImageTextToText, get_cosine_schedule_with_warmup
from peft import PeftModel
base_model = AutoModelForImageTextToText.from_pretrained(
    cfg.model_id, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
)
for p in base_model.parameters(): p.requires_grad = False
base_model.config.use_cache = False
model = PeftModel.from_pretrained(base_model, str(RUN_B_DIR), is_trainable=True)
model.gradient_checkpointing_enable()
if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,} (continuing from Run B)")
assert trainable <= 5_000_000

# ── 6) Optim + cosine schedule ───────────────────────────────────────────
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
total_update_steps = (len(train_loader) // cfg.grad_accum) * cfg.epochs
warmup_steps = max(1, int(total_update_steps * cfg.warmup_ratio))
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)
print(f"total_updates={total_update_steps}  warmup={warmup_steps}  lr={cfg.lr}")

# ── 7) Best-mirror save helper ───────────────────────────────────────────
def _save_best_c(acc, step):
    global best_acc, best_step
    if acc > best_acc:
        best_acc, best_step = acc, step
        model.save_pretrained(cfg.adapter_dir)
        processor.save_pretrained(cfg.adapter_dir)
        for f in Path(cfg.adapter_dir).glob("*"):
            if f.is_file(): shutil.copy2(f, RUN_C_DIR / f.name)
        print(f"  ★ new best ({acc*100:.2f}%) — mirrored to Drive: {RUN_C_DIR}")

# ── 8) Train ─────────────────────────────────────────────────────────────
history = {"step":[], "loss":[], "eval_step":[], "eval_acc":[]}
best_acc, best_step = -1.0, -1
update_step = micro_step = 0
running_loss = 0.0; loss_count = 0
t0 = time.time()

for epoch in range(cfg.epochs):
    print(f"\n=== Phase 2 CoT  Epoch {epoch+1}/{cfg.epochs} ===")
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
                ev = evaluate(model, val_ds, max_samples=None)   # full monitor set (n=420)
                history["eval_step"].append(update_step); history["eval_acc"].append(ev["accuracy"])
                print(f"  ↪ monitor acc (n={ev['n']}): {ev['accuracy']*100:.2f}%")
                _save_best_c(ev["accuracy"], update_step); model.train()
    ev = evaluate(model, val_ds, max_samples=None)
    history["eval_step"].append(update_step); history["eval_acc"].append(ev["accuracy"])
    print(f"--- end epoch {epoch+1} monitor acc (n={ev['n']}): {ev['accuracy']*100:.2f}%")
    _save_best_c(ev["accuracy"], update_step)

print(f"\n✅ Phase 2 CoT done in {(time.time()-t0)/60:.1f} min — best monitor acc = {best_acc*100:.2f}%")

# ── 9) Save meta + history to Drive ──────────────────────────────────────
(RUN_C_DIR / "meta.json").write_text(json.dumps({
    "phase": "2_CoT",
    "started_from": str(RUN_B_DIR),
    "seed": SEED_C, "img_size": cfg.img_size, "epochs": cfg.epochs, "lr": cfg.lr,
    "best_monitor_acc": float(best_acc), "best_step": int(best_step),
    "trainable_params": int(trainable),
    "elapsed_min": round((time.time()-t0)/60, 1),
    "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "training_data": {
        "n_train_orig": int(len(train_df)),
        "n_val_added":  int(len(val_train)),
        "monitor_n":    int(len(val_monitor)),
    },
}, indent=2))
(RUN_C_DIR / "history.json").write_text(json.dumps(history))
print(f"saved meta + history to {RUN_C_DIR}")

# ── 10) Disconnect runtime ───────────────────────────────────────────────
print("Sleeping 60s to flush Drive…")
time.sleep(60)
try:
    from google.colab import runtime
    runtime.unassign()
except Exception as e:
    print("runtime.unassign failed:", e); os._exit(0)
