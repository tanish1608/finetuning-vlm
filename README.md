# Pixels to Predictions — ScienceQA VLM Fine-Tuning

This repository contains the training and inference code for the "Pixels to Predictions" challenge, a visual multiple-choice question answering (ScienceQA) task. The goal is to fine-tune `HuggingFaceTB/SmolVLM-500M-Instruct` while strictly staying under **5 million trainable parameters**.

## 🚀 tl;dr Next Steps
1. **Train:** Run `DL_FINAL_Train.ipynb` (or `train-notebook-tests/mega_run_G_cell.py`) end-to-end on an A100 Colab to produce a LoRA adapter inside `outputs_train/adapter/`. 
2. **Review Validation Results:** Once training finishes, review the `val_predictions.csv` it outputs to identify the hardest subjects.
3. **Hard-Negative Mining:** Use `train-notebook-tests/run_hard_curriculum.py` to upsample the difficult examples into a `train_hard.csv` and execute a final targeted training run.
4. **Inference:** Point the inference script (e.g., `infer-notebook-tests/infer_ocr_aware.py`) to your adapter directory to generate cross-validation metrics or `submission.csv`.

## 🧠 Core Strategy & Tactics
This project employs several strategies to maximize accuracy within the tight <5M parameter bounds:

1. **Multiple-Choice Log-Likelihood Scoring:** We don't call `.generate()`. Instead, we compute log-likelihoods of the specific option tokens (A, B, C, etc.) over a masked prompt. This provides deterministic predictions, avoids verbose hallucination, and offers sharper cross-entropy loss signals.
2. **Rank-Restricted LoRA:** We train exclusively on the language model's attention projections keeping rank bounds (e.g., `r=16, alpha=32`) low enough to assert `< 5,000,000` trainable parameters. The vision tower is intentionally frozen.
3. **Choice-Shuffle Augmentation:** We dynamically randomize the `choices` list per epoch alongside mutating the target letter. This eliminates prior positional bias (e.g., "A" being the answer disproportionately in the dataset).
4. **Test-Time Augmentation (TTA):** During inference, we evaluate multiple shuffled permutations of the options per question, average the logits, and project back to the original index. This reliably yields +1 to +3 accuracy points.
5. **Hard-Negative Mining Curriculum:** We identify subjects the model fails on and upsample them in the dataset prior to final epochs. By duplicating these rows (e.g., complex Physics diagrams) 3x-4x, we dramatically increase exposure to the model's weak points.
6. **OCR-Aware Prompting:** Integrates PaddleOCR and EasyOCR caching directly as text context embedded into the user prompt, aiding the small VLM feature extractor on dense text diagrams.
7. **Ensembling:** Combining weights or logits from runs trained with vs. without OCR, or via different OCR sources (PaddleOCR vs EasyOCR).

## 📁 Key Files
- `DL_FINAL_Train.ipynb` / `DL_FINAL_Train V2.ipynb` - The primary training notebooks.
- `train-notebook-tests/` - Python scripts for various training configurations (e.g., `mega_run_G_cell.py` using PaddleOCR).
- `infer-notebook-tests/` - Standalone inference scripts including `infer_ocr_aware.py` handling TTA and ensembling.
- `STRATEGY.md` - In-depth breakdown of hyper-parameters, things to try, and things to avoid.

## ⚙️ Requirements & Constraints
- `HuggingFaceTB/SmolVLM-500M-Instruct` is the only allowed checkpoint.
- **< 5 M trainable parameters** limit (enforced programmatically in the notebook).
- Only the provided competition data is used (no external datasets).
- Inference runs completely offline.
