# ═══════════════════════════════════════════════════════════════════════════
# OCR PREPROCESSING — extract text from every image in train + val + test
# Run once. Saves a JSON keyed by row id at OCR_JSON.
# Resumable: re-running picks up where it left off (skips already-processed ids).
# Expected runtime on A100: 50–80 min for ~10k images.
# ═══════════════════════════════════════════════════════════════════════════
import os, json, time, gc
from pathlib import Path
import pandas as pd
import torch

# ── Install EasyOCR (GPU-enabled, pure Python) ──
get_ipython().system("pip install -q easyocr")

# ── Paths ────────────────────────────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except ImportError: pass

DATA_DIR  = Path("/content/data")                                          # ⚠️ edit if different
DRIVE_OUT = Path("/content/drive/MyDrive/scienceqa_run_final_overnight")
OCR_JSON  = DRIVE_OUT / "ocr_text.json"
DRIVE_OUT.mkdir(parents=True, exist_ok=True)
SAVE_EVERY = 200          # checkpoint to Drive every N images

# ── Resume from existing JSON if present ─────────────────────────────────
ocr_by_id = {}
if OCR_JSON.exists():
    ocr_by_id = json.loads(OCR_JSON.read_text())
    print(f"resuming — already have OCR for {len(ocr_by_id):,} images")

# ── Build the worklist: all unique (id, image_path) across the three CSVs ─
def _load(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df[["id", "image_path"]]
all_rows = pd.concat([_load("train"), _load("val"), _load("test")], ignore_index=True)
all_rows = all_rows.drop_duplicates("id").reset_index(drop=True)
todo = all_rows[~all_rows["id"].isin(ocr_by_id)].reset_index(drop=True)
print(f"{len(all_rows):,} total images, {len(todo):,} still to process")

# ── EasyOCR reader (GPU) ─────────────────────────────────────────────────
import easyocr
reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
print("EasyOCR ready, GPU =", torch.cuda.is_available())

# ── Process loop with periodic Drive save ────────────────────────────────
def extract(path):
    try:
        # readtext returns list of (bbox, text, conf)
        results = reader.readtext(str(path), detail=1, paragraph=False)
        # keep medium+ confidence, join, truncate
        text = " ".join(r[1] for r in results if r[2] >= 0.3)
        return text[:600]      # truncate per-image to keep prompt size reasonable
    except Exception:
        return ""

t0 = time.time()
for i, row in todo.iterrows():
    p = DATA_DIR / row["image_path"]
    if not p.exists():
        ocr_by_id[row["id"]] = ""
        continue
    ocr_by_id[row["id"]] = extract(p)
    if (i + 1) % SAVE_EVERY == 0:
        OCR_JSON.write_text(json.dumps(ocr_by_id))
        rate = (i+1) / (time.time() - t0)
        eta_min = (len(todo) - (i+1)) / rate / 60
        print(f"  {i+1:>5d}/{len(todo)}  rate={rate:.1f} img/s  ETA={eta_min:.1f} min  "
              f"sample text={ocr_by_id[row['id']][:60]!r}")

# Final flush
OCR_JSON.write_text(json.dumps(ocr_by_id))

# ── Sanity check ─────────────────────────────────────────────────────────
n_with = sum(1 for v in ocr_by_id.values() if v.strip())
n_total = len(ocr_by_id)
print(f"\n✅ OCR done. {n_with:,}/{n_total:,} images produced text "
      f"({n_with/n_total*100:.1f}%)  in {(time.time()-t0)/60:.1f} min")
print("saved to:", OCR_JSON)

# Show 5 random samples so you can eyeball quality
import random
random.seed(0)
for sid in random.sample(list(ocr_by_id.keys()), min(5, len(ocr_by_id))):
    print(f"\n--- {sid} ---")
    print(ocr_by_id[sid][:300] or "(empty)")

# Free OCR model so training can use the GPU
del reader; gc.collect(); torch.cuda.empty_cache()
print("\nGPU cleared. Now proceed to the mega training cell.")
