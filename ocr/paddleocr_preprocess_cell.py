# ═══════════════════════════════════════════════════════════════════════════
# PADDLEOCR PREPROCESSING — extracts text with better quality than EasyOCR.
# Especially helpful for Punnett squares, tables, and labeled diagrams.
# Runs on GPU. Resumable — re-running skips already-processed ids.
# Expected runtime: ~50-80 min on A100 80GB for ~10k images.
# Saves: ocr_text_paddle.json (separate from the EasyOCR cache so both exist)
# ═══════════════════════════════════════════════════════════════════════════
import os, json, time, gc
from pathlib import Path
import pandas as pd
import torch

# ── Install PaddleOCR ─────────────────────────────────────────────────────
get_ipython().system("pip install -q paddlepaddle-gpu==3.0.0b0 paddleocr 2>/dev/null || pip install -q paddlepaddle paddleocr")

# ── Drive ─────────────────────────────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except ImportError: pass

DATA_DIR  = Path("/content/data")
DRIVE_OUT = Path("/content/drive/MyDrive/scienceqa_run_final_overnight")
OCR_JSON  = DRIVE_OUT / "ocr_text_paddle.json"   # ← separate file
DRIVE_OUT.mkdir(parents=True, exist_ok=True)
SAVE_EVERY = 200

# ── Resume support ───────────────────────────────────────────────────────
ocr_by_id = {}
if OCR_JSON.exists():
    ocr_by_id = json.loads(OCR_JSON.read_text())
    print(f"resuming — {len(ocr_by_id):,} already done")

def _load(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df[["id", "image_path"]]
all_rows = pd.concat([_load("train"), _load("val"), _load("test")], ignore_index=True).drop_duplicates("id").reset_index(drop=True)
todo = all_rows[~all_rows["id"].isin(ocr_by_id)].reset_index(drop=True)
print(f"{len(all_rows):,} total | {len(todo):,} to process")

# ── PaddleOCR init ───────────────────────────────────────────────────────
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=torch.cuda.is_available())
print("PaddleOCR ready, GPU =", torch.cuda.is_available())

def extract(path):
    try:
        result = ocr.ocr(str(path), cls=True)
        if not result or not result[0]:
            return ""
        # result[0] is list of [bbox, (text, conf)]
        # Sort top-to-bottom, then left-to-right by bbox center
        items = []
        for line in result[0]:
            if line is None: continue
            bbox, (text, conf) = line[0], line[1]
            if conf < 0.4: continue        # confidence filter (slightly stricter than EasyOCR's 0.3)
            cy = sum(p[1] for p in bbox) / 4
            cx = sum(p[0] for p in bbox) / 4
            items.append((cy, cx, text))
        items.sort(key=lambda x: (round(x[0] / 20), x[1]))   # bin rows by ~20px, then sort by x
        text = " ".join(t for _,_,t in items)
        return text[:600]
    except Exception as e:
        return ""

# ── Process loop ─────────────────────────────────────────────────────────
t0 = time.time()
for i, row in todo.iterrows():
    p = DATA_DIR / row["image_path"]
    if not p.exists():
        ocr_by_id[row["id"]] = ""; continue
    ocr_by_id[row["id"]] = extract(p)
    if (i + 1) % SAVE_EVERY == 0:
        OCR_JSON.write_text(json.dumps(ocr_by_id))
        rate = (i+1) / (time.time() - t0)
        eta_min = (len(todo) - (i+1)) / rate / 60
        print(f"  {i+1:>5d}/{len(todo)}  rate={rate:.1f} img/s  ETA={eta_min:.1f} min  "
              f"sample={ocr_by_id[row['id']][:80]!r}")

OCR_JSON.write_text(json.dumps(ocr_by_id))

# ── Sanity ───────────────────────────────────────────────────────────────
n_with = sum(1 for v in ocr_by_id.values() if v.strip())
print(f"\n✅ done. {n_with:,}/{len(ocr_by_id):,} have text ({n_with/len(ocr_by_id)*100:.1f}%)  in {(time.time()-t0)/60:.1f} min")

# Compare a few against the EasyOCR cache so you can see the quality diff
import random
ez_path = DRIVE_OUT / "ocr_text.json"
if ez_path.exists():
    ez = json.loads(ez_path.read_text())
    print("\n— Quality check vs. EasyOCR (5 random samples) —")
    random.seed(0)
    for sid in random.sample(list(ocr_by_id.keys()), 5):
        print(f"\n{sid}")
        print(f"  EasyOCR : {ez.get(sid, '')[:200]!r}")
        print(f"  Paddle  : {ocr_by_id[sid][:200]!r}")

del ocr; gc.collect(); torch.cuda.empty_cache()
print("\nGPU cleared. Now run mega_run_G_cell.py for training.")
