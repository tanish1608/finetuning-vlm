# ─────────────────────────────────────────────────────────────────────────────
# Paste this as a NEW CELL directly below your training-loop cell.
# When training finishes, Colab runs it automatically -> copies the best
# adapter + plots to Drive, then disconnects the runtime so you stop burning
# A100 credits while you sleep.
# ─────────────────────────────────────────────────────────────────────────────
import os, json, shutil, time, datetime
from pathlib import Path

# 1) Mount Drive (no-op if already mounted)
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    print("✅ Drive mounted")
except ImportError:
    print("Not in Colab — skipping Drive mount")

# 2) Where in Drive to put everything (EDIT IF YOU WANT)
DRIVE_ROOT = Path("/content/drive/MyDrive/scienceqa_run")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

# 3) Copy the best LoRA adapter
src_adapter = Path(cfg.adapter_dir)
dst_adapter = DRIVE_ROOT / "adapter"
if dst_adapter.exists():
    shutil.rmtree(dst_adapter)
shutil.copytree(src_adapter, dst_adapter)
adapter_size_mb = sum(f.stat().st_size for f in dst_adapter.rglob("*") if f.is_file()) / 1e6
print(f"✅ Copied adapter ({adapter_size_mb:.1f} MB) → {dst_adapter}")

# 4) Copy plots + val predictions + anything else in out_dir
copied = 0
for f in Path(cfg.out_dir).glob("*"):
    if f.is_file():
        shutil.copy2(f, DRIVE_ROOT / f.name)
        copied += 1
print(f"✅ Copied {copied} extra files (plots, csvs)")

# 5) Drop a finish marker so you know what state you're in when you wake up
finish_meta = {
    "finished_at"     : datetime.datetime.now().isoformat(timespec="seconds"),
    "best_val_acc"    : float(best_acc),
    "best_step"       : int(best_step),
    "trainable_params": int(trainable),
    "epochs_planned"  : int(cfg.epochs),
    "drive_path"      : str(DRIVE_ROOT),
}
(DRIVE_ROOT / "finish_meta.json").write_text(json.dumps(finish_meta, indent=2))
print("\n=== finish_meta.json ===")
print(json.dumps(finish_meta, indent=2))

# 6) Give Drive a moment to actually flush, then kill the runtime
print("\nSleeping 60s to let Drive fully flush…")
time.sleep(60)

print("Disconnecting runtime to stop credit burn. 💤")
try:
    from google.colab import runtime
    runtime.unassign()
except Exception as e:
    print("runtime.unassign() failed, falling back to process kill:", e)
    os._exit(0)
