# Pixels to Predictions — Strategy & Run Plan

This document explains the approach used in `train.ipynb` and `infer.ipynb`, why each piece is there, and the order in which to iterate if you want to climb the leaderboard.

## TL;DR

1. Run `train.ipynb` end-to-end on an A100 Colab. It saves a LoRA adapter to `outputs_train/adapter/`.
2. Set `ADAPTER_DIR` in `infer.ipynb` to that path and run it. It writes `submission.csv`.
3. Submit. Then iterate using the "What to try next" list below.

## Approach

**Base model.** `HuggingFaceTB/SmolVLM-500M-Instruct` — required by the rules.

**Fine-tuning.** LoRA on attention projections of the language model only (`q_proj, k_proj, v_proj, o_proj`), `r=16`, `alpha=32`, dropout `0.05`. The vision tower stays frozen — fewer params, less overfitting risk, and SigLIP is already strong on natural images.

**Parameter budget.** With these settings the trainable count lands around **3.5–4 M** parameters, well under the 5 M cap. The notebook asserts this and will fail loudly if you push it over.

**Loss target.** During training, the assistant turn is exactly one letter (`A`, `B`, …). The collator masks every token in the prompt with `-100` and only computes cross-entropy on that single answer-letter token. This makes the gradient signal very sharp and prevents wasted capacity on rephrasing the question.

**Inference scoring.** We do **not** call `model.generate`. Instead we pass the prompt with `add_generation_prompt=True`, take the next-token logits, and pick the argmax over the candidate-letter token IDs. This is what the brief calls "multiple-choice log-likelihood." It is faster, deterministic, and strictly more accurate than parsing free-form generations.

**Choice-shuffle augmentation (training).** On every epoch, every example, we randomly permute `choices` and re-target the answer letter accordingly. The training data has positional bias (`A` is correct ~35 % of the time); without this, the model would learn position priors instead of reading the options.

**Test-time augmentation (inference).** `infer.ipynb` runs `K` permutations of the choice list per example, projects the logits back to the original index space, and averages. `K=4` is the default — costs 4× compute, typically worth +1 to +3 accuracy points on small VLMs.

**Prompt construction.** SmolVLM's chat template via `processor.apply_chat_template`. The user turn carries: image → optional truncated lecture (≤ 1200 chars) → optional truncated hint (≤ 600 chars) → the question → labelled choices → an instruction "Answer with a single letter only." The assistant turn during training is just the letter.

## Key hyper-parameters (defaults in `CFG`)

| param | default | reason |
|---|---|---|
| `img_size` | 384 | SmolVLM tile; preserves diagram detail without blowing VRAM |
| `lora_r / alpha` | 16 / 32 | proven sweet spot; alpha = 2·r |
| `epochs` | 2 | usually enough; bump to 3–4 if val acc still climbing |
| `micro_bsz × grad_accum` | 4 × 4 = 16 | fits A100; effective batch 16 is good for AdamW |
| `lr` | 2e-4 | LoRA-typical; use 1e-4 for longer runs |
| `warmup_ratio` | 0.05 | cosine schedule with short warmup |
| `max_context_chars` | 1200 | truncates long lectures; tune if you see them being too aggressive |
| `TTA_PERMUTATIONS` | 4 | inference-time |

## Validation strategy

- Mid-training eval every 250 update steps on 600-sample val subset (fast feedback).
- End-of-epoch eval on 1500 val samples.
- After training, full val eval with **per-subject and per-grade accuracy breakdown** — saved as `val_predictions.csv`. Look at this carefully: that's where you find your next win.

## Things to try next (rough order of expected ROI)

1. **More epochs at lower LR.** `epochs=4, lr=1e-4` — single biggest free win if the loss is still decreasing.
2. **Higher image resolution.** Set `img_size=512`. Helps especially on grade-school diagram questions.
3. **Add MLP LoRA with smaller r.** Attention `r=8` + MLP (`gate_proj, up_proj, down_proj`) `r=4` keeps you under 5 M and adds capacity.
4. **Subject-targeted curriculum.** From `val_predictions.csv`, find the worst subject. Up-weight those rows in a second epoch (e.g. duplicate them in the dataframe).
5. **Increase TTA.** Bump `TTA_PERMUTATIONS` from 4 to 8. Doubles inference time but reliably worth half a point.
6. **Two-seed ensemble.** Run `train.ipynb` twice with `SEED=42` and `SEED=7`. In `infer.ipynb`, load adapter A, score, save logits; load adapter B, score, save logits; average. Worth 1–2 points usually.
7. **Hard-negative mining.** Take examples the v1 model gets wrong on val, append them to train (with shuffled choices), train one more epoch.

## Things to NOT do

- Don't switch to `model.generate` for predictions — log-likelihood scoring strictly dominates.
- Don't add LoRA to the vision tower; the param budget is too tight and the gain is small.
- Don't right-pad at inference — left-padding is required so position `-1` is the actual next-token slot for every row.
- Don't lengthen the prompt past ~1500 tokens; SmolVLM's image tokens already eat a lot of context.
- Don't include external data — competition rule.

## Files this run produces

```
outputs_train/
├── adapter/               # LoRA weights — load this from infer.ipynb
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── train_meta.json
│   └── (tokenizer + processor files)
├── eda_overview.png
├── train_curves.png
└── val_predictions.csv
submission.csv             # produced by infer.ipynb at the project root
```
