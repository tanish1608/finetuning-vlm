0# ═══════════════════════════════════════════════════════════════════════════
# SECOND ADAPTER (seed=7, img_size=512) + Drive backup + auto-disconnect
# Paste as ONE cell directly below your finished training cell.
# Assumes you've already executed all train.ipynb cells through cell 12 and
# the seed-42 adapter is sitting at cfg.adapter_dir (default outputs_train/adapter).
# ═══════════════════════════════════════════════════════════════════════════
import os, gc, json, time, random, shutil, datetime
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW

# ── 1) Mount Drive + back up the seed-42 adapter immediately ──────────────
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    print("✅ Drive mounted")
except ImportError:
    print("ℹ️  Not in Colab")

DRIVE_ROOT = Path("/content/drive/MyDrive/scienceqa_run")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

dst_a = DRIVE_ROOT / "seed42_res384"
if dst_a.exists(): shutil.rmtree(dst_a)
shutil.copytree(cfg.adapter_dir, dst_a)
(dst_a / "meta.json").write_text(json.dumps({
    "seed": 42, "img_size": cfg.img_size, "best_val_acc": float(best_acc),
    "best_step": int(best_step), "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
}, indent=2))
print(f"✅ Run A backed up to {dst_a}")

# ── 2) Free GPU memory from Run A ─────────────────────────────────────────
try:
    del model, optimizer, scheduler, trainable_params, base_model
except NameError:
    pass
gc.collect(); torch.cuda.empty_cache()
print(f"GPU free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

# ── 3) Reconfigure for Run B ──────────────────────────────────────────────
SEED2 = 7
random.seed(SEED2); np.random.seed(SEED2); torch.manual_seed(SEED2); torch.cuda.manual_seed_all(SEED2)

cfg.epochs       = 3
cfg.img_size     = 512                                  # higher res for diagrams
cfg.adapter_dir  = Path("outputs_train/adapter_seed7"); cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
local_drive_b    = DRIVE_ROOT / "seed7_res512"
local_drive_b.mkdir(parents=True, exist_ok=True)

# ── 4) Rebuild datasets + loader at the new resolution ────────────────────
train_ds = ScienceQADataset(train_df, cfg.data_dir, img_size=cfg.img_size,
                            is_train=True, shuffle_choices=cfg.shuffle_choices_train)
val_ds   = ScienceQADataset(val_df,   cfg.data_dir, img_size=cfg.img_size, is_train=False)
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=cfg.micro_bsz, shuffle=True,
    collate_fn=train_collate, num_workers=0, pin_memory=True, drop_last=True,
)
print(f"Run B datasets ready (img_size={cfg.img_size})  steps/epoch={len(train_loader)}")

# ── 5) Reload BASE model fresh (no carry-over from seed-42 LoRA) ──────────
from transformers import AutoModelForImageTextToText, get_cosine_schedule_with_warmup
from peft import get_peft_model
base_model = AutoModelForImageTextToText.from_pretrained(
    cfg.model_id, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
)
for p in base_model.parameters(): p.requires_grad = False
base_model.config.use_cache = False
model = get_peft_model(base_model, lora_cfg)        # same LoRA topology as Run A
model.gradient_checkpointing_enable()
if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params (Run B): {trainable:,}")
assert trainable <= 5_000_000

# ── 6) Optim + cosine schedule for Run B's horizon ────────────────────────
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
total_update_steps = (len(train_loader) // cfg.grad_accum) * cfg.epochs
warmup_steps = max(1, int(total_update_steps * cfg.warmup_ratio))
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)
print(f"Run B  total_updates={total_update_steps}  warmup={warmup_steps}")

# ── 7) Helper that mirrors best adapter to Drive on every improvement ────
def _save_best(acc, step):
    global best_acc, best_step
    if acc > best_acc:
        best_acc, best_step = acc, step
        model.save_pretrained(cfg.adapter_dir)
        processor.save_pretrained(cfg.adapter_dir)
        for f in Path(cfg.adapter_dir).glob("*"):
            if f.is_file(): shutil.copy2(f, local_drive_b / f.name)
        print(f"  ★ new best ({acc*100:.2f}%) — mirrored to Drive: {local_drive_b}")

# ── 8) Training loop (inlined) ────────────────────────────────────────────
history = {"step": [], "loss": [], "eval_step": [], "eval_acc": []}
best_acc, best_step = -1.0, -1
update_step = micro_step = 0
running_loss = 0.0; loss_count = 0
t0 = time.time()

for epoch in range(cfg.epochs):
    print(f"\n=== Run B (seed=7) Epoch {epoch+1}/{cfg.epochs} ===")
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
                ev = evaluate(model, val_ds, max_samples=cfg.eval_max_samples)
                history["eval_step"].append(update_step); history["eval_acc"].append(ev["accuracy"])
                print(f"  ↪ val acc (n={ev['n']}): {ev['accuracy']*100:.2f}%")
                _save_best(ev["accuracy"], update_step); model.train()
    ev = evaluate(model, val_ds, max_samples=min(len(val_ds), 1500))
    history["eval_step"].append(update_step); history["eval_acc"].append(ev["accuracy"])
    print(f"--- end epoch {epoch+1} val acc (n={ev['n']}): {ev['accuracy']*100:.2f}%")
    _save_best(ev["accuracy"], update_step)

print(f"\n✅ Run B done in {(time.time()-t0)/60:.1f} min — best val = {best_acc*100:.2f}%")

# ── 9) Final meta + summary on Drive ──────────────────────────────────────
(local_drive_b / "meta.json").write_text(json.dumps({
    "seed": 7, "img_size": 512, "epochs": cfg.epochs,
    "best_val_acc": float(best_acc), "best_step": int(best_step),
    "trainable_params": int(trainable),
    "elapsed_min": round((time.time()-t0)/60, 1),
    "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
}, indent=2))
(local_drive_b / "history.json").write_text(json.dumps(history))

summary = {
    "run_A": str(dst_a),
    "run_B": str(local_drive_b),
    "run_B_best_val_acc": float(best_acc),
    "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
}
(DRIVE_ROOT / "OVERNIGHT_SUMMARY.json").write_text(json.dumps(summary, indent=2))
print("\n=== OVERNIGHT SUMMARY ===")
print(json.dumps(summary, indent=2))

# ── 10) Disconnect to stop credit burn ────────────────────────────────────
print("\nSleeping 60s to flush Drive…")
time.sleep(60)
print("Disconnecting runtime. 💤")
try:
    from google.colab import runtime
    runtime.unassign()
except Exception as e:
    print("runtime.unassign failed:", e)
    os._exit(0)
