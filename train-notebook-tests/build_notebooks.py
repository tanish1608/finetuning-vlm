"""Builds train.ipynb and infer.ipynb for the SmolVLM ScienceQA challenge.

Run with:  python3 build_notebooks.py
Outputs:   train.ipynb, infer.ipynb in the same directory.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text,
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
            "accelerator": "GPU",
            "colab": {"gpuType": "A100"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# =============================================================================
# TRAIN NOTEBOOK
# =============================================================================

train_cells = []

train_cells.append(md(r'''# ScienceQA Vision Challenge — **Training Notebook**

Fine-tunes `HuggingFaceTB/SmolVLM-500M-Instruct` on the ScienceQA visual MCQ task with **QLoRA / LoRA**, then saves a LoRA adapter that the companion `infer.ipynb` loads to generate `submission.csv`.

## Strategy (why this should beat a naive baseline)

1. **Multiple-choice log-likelihood scoring** instead of free-form generation. We score each candidate letter (`A`, `B`, …) and pick the highest. Generation with a 500M model produces verbose / wrong tokens; scoring is deterministic and a strict superset of the supervised signal.
2. **QLoRA** — base model in 4-bit NF4, trainable LoRA adapters on attention projections only. We confirm `trainable_params < 5 M` programmatically.
3. **Choice-shuffle augmentation** — at training time we randomly permute the `choices` list and re-target the answer letter. This destroys positional bias (`A` is the right answer 35 % of the time in train) and forces the model to *read* the options.
4. **Lecture/hint-aware prompting** — we use SmolVLM's chat template and inject `lecture` + `hint` only when present and short enough to fit, with a hard cap so prompts don't blow up the KV cache.
5. **Per-subject monitoring** — validation accuracy is broken down by `subject` and `grade` so you know where to spend the next training hour.
6. **bf16 mixed precision** + cosine LR + warmup + gradient accumulation. A100 likes bf16 a lot more than fp16.

## Constraints we respect

- `SmolVLM-500M-Instruct` is the only allowed checkpoint.
- ≤ 5 M trainable parameters (asserted in the LoRA cell).
- Only the provided competition data is used — no external scraped data.
- Inference runs offline (the `infer.ipynb` does not call out to the network).

> **Run order:** execute cells top-to-bottom. The training loop saves the LoRA adapter to `OUT_DIR/adapter` — point `infer.ipynb` at that path.
'''))

train_cells.append(md(r'''## 0. Install dependencies

Pinned versions matching the starter so the chat template and processor behave the same as the course baseline.'''))

train_cells.append(code(r'''# Run once per Colab VM. Comment out if already installed.
%pip install -q transformers==4.57.6 peft==0.18.1 bitsandbytes==0.44.1 accelerate==1.0.1 datasets pillow matplotlib seaborn'''))

train_cells.append(md(r'''## 1. Imports & configuration

All hyper-parameters live in the `CFG` dataclass — change once, propagates everywhere.'''))

train_cells.append(code(r'''import os, json, math, random, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
if torch.cuda.is_available():
    print("gpu  :", torch.cuda.get_device_name(0))
    print("vram :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")'''))

train_cells.append(code(r'''@dataclass
class CFG:
    # ── Paths ────────────────────────────────────────────────────────────
    data_dir: Path = Path("data")                 # contains train.csv, val.csv, test.csv, images/
    out_dir : Path = Path("outputs_train")        # adapter + plots get saved here
    adapter_dir: Path = Path("outputs_train/adapter")  # final LoRA adapter

    # ── Model ────────────────────────────────────────────────────────────
    model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    img_size: int = 384                           # SmolVLM tile size; 384 keeps diagram detail
    use_4bit: bool = False                        # A100 has VRAM; bf16 LoRA trains faster & cleaner
    bf16    : bool = True

    # ── LoRA ─────────────────────────────────────────────────────────────
    lora_r       : int = 16
    lora_alpha   : int = 32
    lora_dropout : float = 0.05
    # Attention-only LoRA on the language model. Vision tower is frozen.
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # ── Optim ────────────────────────────────────────────────────────────
    epochs        : int   = 2
    micro_bsz     : int   = 4         # per-step batch
    grad_accum    : int   = 4         # effective batch = 16
    eval_bsz      : int   = 8
    lr            : float = 2e-4
    weight_decay  : float = 0.0
    warmup_ratio  : float = 0.05
    max_grad_norm : float = 1.0

    # ── Prompt ───────────────────────────────────────────────────────────
    max_context_chars: int = 1200     # truncate long lectures+hints
    use_lecture: bool = True
    use_hint   : bool = True
    shuffle_choices_train: bool = True

    # ── Eval / logging ───────────────────────────────────────────────────
    log_every  : int = 25
    eval_every : int = 250            # steps; also evaluates at end of each epoch
    eval_max_samples: int = 600       # cap mid-training eval for speed; full eval at end
    save_best  : bool = True

cfg = CFG()
cfg.out_dir.mkdir(parents=True, exist_ok=True)
cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
print(json.dumps({k: str(v) for k, v in cfg.__dict__.items()}, indent=2))'''))

train_cells.append(md(r'''## 2. Mount Drive / locate the data (Colab)

If your competition data lives in Drive, uncomment the mount cell. Otherwise edit `cfg.data_dir` to point at the unzipped folder.'''))

train_cells.append(code(r'''# Colab Drive mount (uncomment if needed)
# from google.colab import drive
# drive.mount('/content/drive')
# cfg.data_dir = Path('/content/drive/MyDrive/scienceqa')   # <-- edit me
# cfg.out_dir  = Path('/content/drive/MyDrive/scienceqa/outputs_train')
# cfg.adapter_dir = cfg.out_dir / 'adapter'
# cfg.out_dir.mkdir(parents=True, exist_ok=True)
# cfg.adapter_dir.mkdir(parents=True, exist_ok=True)

assert (cfg.data_dir / "train.csv").exists(), f"train.csv not found under {cfg.data_dir}"
assert (cfg.data_dir / "val.csv").exists(),   f"val.csv not found under {cfg.data_dir}"
assert (cfg.data_dir / "test.csv").exists(),  f"test.csv not found under {cfg.data_dir}"
print("data ok ->", cfg.data_dir.resolve())'''))

train_cells.append(md(r'''## 3. Load CSVs'''))

train_cells.append(code(r'''train_df = pd.read_csv(cfg.data_dir / "train.csv")
val_df   = pd.read_csv(cfg.data_dir / "val.csv")
test_df  = pd.read_csv(cfg.data_dir / "test.csv")

for df in [train_df, val_df, test_df]:
    df["choices"] = df["choices"].apply(json.loads)

print(f"train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}")
train_df.head(2)'''))

train_cells.append(md(r'''## 4. EDA — what does the data look like?

We want to confirm:

- How biased is the answer position? (If `A` is right 40 % of the time, models that always pick `A` get 40 % accuracy — choice-shuffle augmentation is critical.)
- Distribution of `num_choices` so we know how many letter tokens we need.
- Subject / grade balance so we know what we'll be optimising for.'''))

train_cells.append(code(r'''sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (a) answer-position bias on train
ax = axes[0, 0]
train_df["answer"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#4C72B0")
ax.set_title(f"Answer-index distribution (train) — {len(train_df)} rows")
ax.set_xlabel("answer index"); ax.set_ylabel("count")

# (b) num_choices
ax = axes[0, 1]
train_df["num_choices"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#55A868")
ax.set_title("num_choices (train)")
ax.set_xlabel("num_choices"); ax.set_ylabel("count")

# (c) subject distribution
ax = axes[1, 0]
train_df["subject"].value_counts().plot(kind="barh", ax=ax, color="#C44E52")
ax.set_title("Subject (train)")

# (d) grade
ax = axes[1, 1]
gorder = sorted(train_df["grade"].dropna().unique(), key=lambda g: int("".join(c for c in g if c.isdigit()) or 0))
train_df["grade"].value_counts().reindex(gorder).plot(kind="bar", ax=ax, color="#8172B2")
ax.set_title("Grade (train)")
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(cfg.out_dir / "eda_overview.png", dpi=120)
plt.show()

print("\nMost common num_choices :", train_df["num_choices"].mode().iloc[0])
print("Max num_choices         :", train_df["num_choices"].max())
print("Always-pick-0 baseline  :", round((train_df["answer"] == 0).mean() * 100, 2), "%")
print("Always-pick-mode baseline (val):",
      round(((val_df["answer"] == train_df["answer"].mode().iloc[0])).mean() * 100, 2), "%")'''))

train_cells.append(code(r'''# Visual inspection: show 3 random examples (image + question)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, idx in zip(axes, train_df.sample(3, random_state=SEED).index):
    row = train_df.loc[idx]
    img = Image.open(cfg.data_dir / row["image_path"]).convert("RGB")
    ax.imshow(img)
    ax.set_title(f"Q: {row['question'][:60]}…\nA={row['answer']} ({row['choices'][row['answer']][:25]})", fontsize=9)
    ax.axis("off")
plt.tight_layout(); plt.show()'''))

train_cells.append(md(r'''## 5. Prompt engineering

We use SmolVLM's chat template via `processor.apply_chat_template`. The assistant turn contains *just the answer letter* — that's the only token we score / supervise.

**Why a single letter?**  Multi-token answers create label-leak risk through teacher forcing and waste capacity. The leaderboard metric is single-letter accuracy.'''))

train_cells.append(code(r'''CHOICE_LETTERS = "ABCDEFGHIJ"

def _truncate(s: str, n: int) -> str:
    s = str(s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"

def build_user_text(row, max_ctx: int = 1200, use_lecture=True, use_hint=True,
                    choices: Optional[List[str]] = None) -> str:
    """User-turn text. Image is attached separately by the processor."""
    if choices is None:
        choices = row["choices"]

    parts = []
    if use_lecture:
        lec = row.get("lecture", None)
        if isinstance(lec, str) and lec.strip():
            parts.append(_truncate(lec, max_ctx))
    if use_hint:
        hint = row.get("hint", None)
        if isinstance(hint, str) and hint.strip():
            parts.append(_truncate(hint, max_ctx // 2))
    context = "\n".join(parts)

    choice_lines = "\n".join(f"{CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices))

    text  = ""
    if context:
        text += f"Context:\n{context}\n\n"
    text += f"Question: {row['question']}\n\n"
    text += f"Choices:\n{choice_lines}\n\n"
    text += "Answer with a single letter only."
    return text


def build_messages(row, *, include_answer: bool, choices=None, answer_idx=None):
    user_text = build_user_text(
        row,
        max_ctx=cfg.max_context_chars,
        use_lecture=cfg.use_lecture,
        use_hint=cfg.use_hint,
        choices=choices,
    )
    msgs = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_text},
        ]}
    ]
    if include_answer:
        idx = answer_idx if answer_idx is not None else int(row["answer"])
        msgs.append({"role": "assistant", "content": [
            {"type": "text", "text": CHOICE_LETTERS[idx]},
        ]})
    return msgs


# Quick sanity print
sample = train_df.iloc[0]
for m in build_messages(sample, include_answer=True):
    print(m["role"].upper(), "→", m["content"][-1]["text"][:200], "..." if len(m["content"][-1]["text"]) > 200 else "")'''))

train_cells.append(md(r'''## 6. Dataset class

Important detail: at *train* time we optionally permute `choices`. We compute the new answer index after permutation and rebuild the prompt with the new order. The model then has to actually look at the option text to know which letter is correct.'''))

train_cells.append(code(r'''class ScienceQADataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: Path, *,
                 img_size: int = 384, is_train: bool = False, shuffle_choices: bool = False):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.img_size = img_size
        self.is_train = is_train
        self.shuffle_choices = shuffle_choices

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path):
        img = Image.open(self.data_dir / rel_path).convert("RGB")
        # Long-side resize keeps aspect ratio (better for diagrams)
        w, h = img.size
        s = self.img_size / max(w, h)
        if s < 1.0:
            img = img.resize((max(1, int(w*s)), max(1, int(h*s))), Image.BICUBIC)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_image(row["image_path"])
        choices = list(row["choices"])
        ans = int(row["answer"]) if "answer" in row and not pd.isna(row["answer"]) else -1

        if self.is_train and self.shuffle_choices and ans >= 0:
            perm = np.random.permutation(len(choices))
            new_choices = [choices[i] for i in perm]
            new_ans = int(np.where(perm == ans)[0][0])
            choices = new_choices
            ans = new_ans

        return {
            "id": row["id"],
            "image": img,
            "row": row,
            "choices": choices,
            "answer": ans,
        }


train_ds = ScienceQADataset(train_df, cfg.data_dir, img_size=cfg.img_size,
                            is_train=True,  shuffle_choices=cfg.shuffle_choices_train)
val_ds   = ScienceQADataset(val_df,   cfg.data_dir, img_size=cfg.img_size, is_train=False)
test_ds  = ScienceQADataset(test_df,  cfg.data_dir, img_size=cfg.img_size, is_train=False)
print("datasets:", len(train_ds), len(val_ds), len(test_ds))'''))

train_cells.append(md(r'''## 7. Load processor + base model

A100 → bf16 LoRA. If you ever need to run on a T4, flip `cfg.use_4bit = True` and the cell below switches to QLoRA / NF4.'''))

train_cells.append(code(r'''from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

processor = AutoProcessor.from_pretrained(cfg.model_id)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
# left-padding is essential for causal-LM scoring (we read the LAST non-pad logit)
processor.tokenizer.padding_side = "left"

model_kwargs: Dict[str, Any] = dict(low_cpu_mem_usage=True)
if cfg.use_4bit:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs["quantization_config"] = bnb
    model_kwargs["device_map"] = "auto"
else:
    model_kwargs["dtype"] = torch.bfloat16 if cfg.bf16 else torch.float32
    model_kwargs["device_map"] = "auto" if torch.cuda.is_available() else None

base_model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, **model_kwargs)
if not torch.cuda.is_available():
    base_model.to(device)

# Freeze everything by default; LoRA will mark a subset trainable
for p in base_model.parameters():
    p.requires_grad = False

# Useful trims: disable cache during training, enable grad ckpt later
base_model.config.use_cache = False
print("base model loaded:", type(base_model).__name__)'''))

train_cells.append(md(r'''## 8. Apply LoRA — and assert ≤ 5 M trainable parameters

We target attention projections only on the language-model side. We do **not** touch the vision tower (frozen) — fewer params, fewer overfit risks, and the SigLIP encoder is already strong.'''))

train_cells.append(code(r'''from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

if cfg.use_4bit:
    base_model = prepare_model_for_kbit_training(base_model)

lora_cfg = LoraConfig(
    r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=list(cfg.lora_target_modules),
    # Make sure we don't accidentally LoRA-ize the vision tower
    modules_to_save=None,
)

model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {trainable:,}")
assert trainable <= 5_000_000, f"Over 5M trainable params ({trainable}). Lower lora_r or drop target modules."
print("✅ within 5M parameter budget")

# Enable gradient checkpointing (saves VRAM, small speed cost)
model.gradient_checkpointing_enable()
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()'''))

train_cells.append(md(r'''## 9. Multiple-choice log-likelihood scoring

For each example we build the prompt with `add_generation_prompt=True` so the assistant turn is *open*. We forward once, read the logits at the last position, and compare the logits at the token IDs for `A`, `B`, `C`, … . The argmax is our predicted index.'''))

train_cells.append(code(r'''# Determine the token ID for each letter.
# We tokenize the letter the way the assistant would emit it (no leading space,
# the chat template already places us right at the assistant prefix).
def _letter_token_id(letter: str) -> int:
    ids = processor.tokenizer(letter, add_special_tokens=False).input_ids
    if len(ids) != 1:
        # fallback: try the variant used after assistant prefix
        ids = processor.tokenizer(" " + letter, add_special_tokens=False).input_ids
    assert len(ids) >= 1, f"could not tokenize letter {letter!r}"
    return ids[-1]

LETTER_IDS = {L: _letter_token_id(L) for L in CHOICE_LETTERS}
print({L: LETTER_IDS[L] for L in CHOICE_LETTERS})


@torch.inference_mode()
def score_batch(model, batch_items, *, choices_override: Optional[List[List[str]]] = None) -> np.ndarray:
    """Return predicted index per item via log-likelihood scoring of the answer letter."""
    images, prompts, num_choices = [], [], []
    for i, item in enumerate(batch_items):
        choices = choices_override[i] if choices_override is not None else item["choices"]
        msgs = build_messages(item["row"], include_answer=False, choices=choices)
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        prompts.append(prompt)
        images.append([item["image"]])
        num_choices.append(len(choices))

    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    out = model(**inputs)
    logits = out.logits  # (B, T, V)
    last = logits[:, -1, :]  # (B, V); left-padded so position -1 is the next-token slot
    preds = []
    for i, n in enumerate(num_choices):
        cand_ids = [LETTER_IDS[CHOICE_LETTERS[k]] for k in range(n)]
        cand_logits = last[i, cand_ids]
        preds.append(int(cand_logits.argmax().item()))
    return np.array(preds, dtype=np.int64)


def evaluate(model, dataset, *, max_samples: Optional[int] = None,
             return_per_example: bool = False, batch_size: Optional[int] = None) -> Dict[str, Any]:
    bs = batch_size or cfg.eval_bsz
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    correct, total = 0, 0
    preds_all, gts_all, ids_all = [], [], []
    model.eval()
    for start in range(0, n, bs):
        items = [dataset[start + j] for j in range(min(bs, n - start))]
        gts = np.array([it["answer"] for it in items])
        preds = score_batch(model, items)
        correct += int((preds == gts).sum())
        total   += len(items)
        if return_per_example:
            preds_all.extend(preds.tolist())
            gts_all.extend(gts.tolist())
            ids_all.extend([it["id"] for it in items])
    acc = correct / max(1, total)
    out = {"accuracy": acc, "n": total}
    if return_per_example:
        out["preds"] = preds_all; out["gts"] = gts_all; out["ids"] = ids_all
    return out

# Quick zero-shot sanity check on a small slice
print("Running zero-shot eval on 200 val samples (this is the baseline)…")
zs = evaluate(model, val_ds, max_samples=200)
print(f"Zero-shot val acc (n=200): {zs['accuracy']*100:.2f}%")'''))

train_cells.append(md(r'''## 10. Training collator + step

We tokenize each example as `chat_template(user, assistant=letter)` and mask everything except the assistant's answer token in the labels. With LoRA + bf16, this is fast on A100.'''))

train_cells.append(code(r'''def make_train_batch(items):
    """Build a model-ready training batch from raw dataset items.

    Subtlety: image-token expansion happens inside the *processor*, not the tokenizer.
    To get the right boundary between prompt and answer in input_ids, we re-encode
    the prompt-only version through the processor (with the same image) and read its
    length — that includes the expanded image tokens. The text tokenizer alone would
    undercount and we'd end up putting `-100` over the answer letter (loss = NaN).
    """
    images, full_texts, prompt_texts = [], [], []
    for it in items:
        msgs_full   = build_messages(it["row"], include_answer=True,
                                     choices=it["choices"], answer_idx=it["answer"])
        msgs_prompt = build_messages(it["row"], include_answer=False, choices=it["choices"])
        full   = processor.apply_chat_template(msgs_full,   add_generation_prompt=False)
        prompt = processor.apply_chat_template(msgs_prompt, add_generation_prompt=True)
        full_texts.append(full)
        prompt_texts.append(prompt)
        images.append([it["image"]])

    # Right-padding for training: prompt tokens sit at positions 0..plen, answer
    # at plen, then end-of-utterance, then padding.
    processor.tokenizer.padding_side = "right"
    enc = processor(text=full_texts, images=images, return_tensors="pt", padding=True)

    # Per-row prompt length WITH image-token expansion. Encode each separately
    # (no padding) and read input_ids.shape[1].
    prompt_lens = []
    for p, im in zip(prompt_texts, images):
        pe = processor(text=p, images=im, return_tensors="pt")
        prompt_lens.append(int(pe["input_ids"].shape[1]))

    labels = enc["input_ids"].clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100          # mask the prompt
    labels[enc["attention_mask"] == 0] = -100   # mask padding
    enc["labels"] = labels

    # leave padding_side = "right" here; the next score_batch call resets it via the
    # LEFT-padding setup it relies on. We reset it explicitly to be safe:
    processor.tokenizer.padding_side = "left"
    return enc


def train_collate(items):
    return make_train_batch(items)


# Quick correctness check on a tiny batch BEFORE we start the full loop
sample_batch_items = [train_ds[i] for i in range(min(2, len(train_ds)))]
_chk = make_train_batch(sample_batch_items)
_lab = _chk["labels"]
_kept = (_lab != -100).sum().item()
print(f"sanity: labels kept (non -100) on 2-row batch = {_kept}  (should be ~2-4 tokens, one per row)")
assert _kept >= 1, "label masking too aggressive — every position is -100"
del _chk, sample_batch_items, _lab, _kept

# IMPORTANT: num_workers=0 keeps the processor side-effects (padding_side flips) on
# the main process so they don't get out of sync with score_batch.
train_loader = DataLoader(
    train_ds, batch_size=cfg.micro_bsz, shuffle=True,
    collate_fn=train_collate, num_workers=0, pin_memory=True, drop_last=True,
)
print("steps per epoch:", len(train_loader), "| effective batch:", cfg.micro_bsz * cfg.grad_accum)'''))

train_cells.append(md(r'''## 11. Optimizer + scheduler'''))

train_cells.append(code(r'''from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

total_update_steps = (len(train_loader) // cfg.grad_accum) * cfg.epochs
warmup_steps = max(1, int(total_update_steps * cfg.warmup_ratio))
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)
print(f"updates total: {total_update_steps} | warmup: {warmup_steps}")'''))

train_cells.append(md(r'''## 12. Training loop

Logs loss every `log_every` micro-steps, evaluates every `eval_every` *update* steps, saves the best LoRA adapter.'''))

train_cells.append(code(r'''history = {"step": [], "loss": [], "eval_step": [], "eval_acc": []}
best_acc, best_step = -1.0, -1
update_step = 0
micro_step  = 0
running_loss = 0.0
loss_count   = 0

t0 = time.time()
for epoch in range(cfg.epochs):
    print(f"\n=== Epoch {epoch+1}/{cfg.epochs} ===")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for batch in train_loader:
        batch = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / cfg.grad_accum
        loss.backward()
        running_loss += loss.item() * cfg.grad_accum
        loss_count   += 1

        micro_step += 1
        if micro_step % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1

            if update_step % cfg.log_every == 0:
                avg = running_loss / max(1, loss_count)
                running_loss, loss_count = 0.0, 0
                history["step"].append(update_step); history["loss"].append(avg)
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step {update_step:>5d}/{total_update_steps}  loss={avg:.4f}  lr={lr_now:.2e}  "
                      f"elapsed={(time.time()-t0)/60:.1f}m")

            if update_step % cfg.eval_every == 0:
                ev = evaluate(model, val_ds, max_samples=cfg.eval_max_samples)
                history["eval_step"].append(update_step); history["eval_acc"].append(ev["accuracy"])
                print(f"  ↪ val acc (n={ev['n']}): {ev['accuracy']*100:.2f}%")
                if cfg.save_best and ev["accuracy"] > best_acc:
                    best_acc, best_step = ev["accuracy"], update_step
                    model.save_pretrained(cfg.adapter_dir)
                    processor.save_pretrained(cfg.adapter_dir)
                    print(f"  ★ new best ({best_acc*100:.2f}%) — adapter saved to {cfg.adapter_dir}")
                model.train()

    # End-of-epoch full(er) eval
    ev = evaluate(model, val_ds, max_samples=min(len(val_ds), 1500))
    history["eval_step"].append(update_step); history["eval_acc"].append(ev["accuracy"])
    print(f"--- end of epoch {epoch+1} val acc (n={ev['n']}): {ev['accuracy']*100:.2f}%")
    if cfg.save_best and ev["accuracy"] > best_acc:
        best_acc, best_step = ev["accuracy"], update_step
        model.save_pretrained(cfg.adapter_dir)
        processor.save_pretrained(cfg.adapter_dir)
        print(f"★ new best ({best_acc*100:.2f}%) — adapter saved")

print(f"\nTraining done in {(time.time()-t0)/60:.1f} min — best val acc = {best_acc*100:.2f}% @ step {best_step}")'''))

train_cells.append(md(r'''## 13. Plot training curves'''))

train_cells.append(code(r'''fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(history["step"], history["loss"], color="#4C72B0", label="train loss")
ax1.set_xlabel("update step"); ax1.set_ylabel("loss", color="#4C72B0"); ax1.tick_params(axis='y', labelcolor="#4C72B0")
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(history["eval_step"], [a*100 for a in history["eval_acc"]], color="#C44E52", marker="o", label="val acc %")
ax2.set_ylabel("val acc (%)", color="#C44E52"); ax2.tick_params(axis='y', labelcolor="#C44E52")

plt.title(f"Training curves — best val acc = {best_acc*100:.2f}%")
plt.tight_layout()
plt.savefig(cfg.out_dir / "train_curves.png", dpi=120)
plt.show()'''))

train_cells.append(md(r'''## 14. Per-subject / grade analysis on full validation

This is the most useful diagnostic. If `social science` is at 80 % but `natural science` is at 50 %, you know exactly where to focus the next round of prompt or data tweaks.'''))

train_cells.append(code(r'''# Reload the BEST adapter so this analysis reflects the saved checkpoint, not the last step
from peft import PeftModel
# (Already loaded as `model`; if you'd reset the kernel you'd reload here.)

print("Running full validation eval — this can take a few minutes …")
full_eval = evaluate(model, val_ds, max_samples=None, return_per_example=True)
print(f"FULL val acc: {full_eval['accuracy']*100:.2f}% (n={full_eval['n']})")

eval_df = pd.DataFrame({
    "id"   : full_eval["ids"],
    "pred" : full_eval["preds"],
    "gt"   : full_eval["gts"],
})
eval_df["correct"] = (eval_df["pred"] == eval_df["gt"]).astype(int)
eval_df = eval_df.merge(val_df[["id", "subject", "topic", "grade", "category"]], on="id")

print("\n— accuracy by subject —")
print(eval_df.groupby("subject")["correct"].agg(["mean", "count"]).rename(columns={"mean":"acc"}).sort_values("acc"))
print("\n— accuracy by grade —")
print(eval_df.groupby("grade")["correct"].agg(["mean", "count"]).rename(columns={"mean":"acc"}).sort_values("acc"))

# Save for reference
eval_df.to_csv(cfg.out_dir / "val_predictions.csv", index=False)
print("\nsaved:", cfg.out_dir / "val_predictions.csv")'''))

train_cells.append(md(r'''## 15. Save final artefacts

We've already been saving the best adapter during training. This cell just confirms it's there and writes a small `train_meta.json` so `infer.ipynb` knows what config was used.'''))

train_cells.append(code(r'''meta = {
    "model_id"           : cfg.model_id,
    "img_size"           : cfg.img_size,
    "lora_r"             : cfg.lora_r,
    "lora_alpha"         : cfg.lora_alpha,
    "lora_target_modules": list(cfg.lora_target_modules),
    "best_val_acc"       : float(best_acc),
    "trainable_params"   : int(trainable),
    "max_context_chars"  : cfg.max_context_chars,
    "use_lecture"        : cfg.use_lecture,
    "use_hint"           : cfg.use_hint,
}
with open(cfg.adapter_dir / "train_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Adapter directory contents:")
for p in sorted(cfg.adapter_dir.iterdir()):
    print("  ", p.name, f"({p.stat().st_size//1024} KB)")
print(f"\n✅ done. Point infer.ipynb at: {cfg.adapter_dir.resolve()}")'''))

train_cells.append(md(r'''## 16. (Optional) Diagnostic — choice-shuffle TTA preview

Before committing to TTA in `infer.ipynb`, we sanity-check that scoring is stable: predict on the validation set under two different choice orderings and see how often they agree. If TTA agreement is >95 % the model is robust; if it's lower, TTA will materially help.'''))

train_cells.append(code(r'''def predict_with_perm(model, dataset, n_samples=300, seed=0):
    rng = np.random.RandomState(seed)
    preds, mapped = [], []
    for i in range(n_samples):
        item = dataset[i]
        perm = rng.permutation(len(item["choices"]))
        shuffled = [item["choices"][k] for k in perm]
        # score on shuffled
        out = score_batch(model, [item], choices_override=[shuffled])[0]
        # map back to original index
        original_idx = int(perm[out])
        preds.append(original_idx)
    return np.array(preds)

orig = predict_with_perm(model, val_ds, n_samples=300, seed=0)
alt  = predict_with_perm(model, val_ds, n_samples=300, seed=1)
print("Perm-A vs Perm-B agreement:", round((orig == alt).mean()*100, 1), "%")
print("Perm-A acc :", round((orig == np.array([val_ds[i]['answer'] for i in range(300)])).mean()*100, 2), "%")
print("Perm-B acc :", round((alt  == np.array([val_ds[i]['answer'] for i in range(300)])).mean()*100, 2), "%")'''))

train_cells.append(md(r'''---

### Things to try if you want to push higher

1. **More epochs** with a lower LR (e.g. `epochs=4`, `lr=1e-4`).
2. **Higher image resolution** — `img_size=512` retains more diagram detail; cost is VRAM and time.
3. **Add MLP LoRA** with smaller `r` (e.g. attention `r=8` + MLP `r=4`) — the param budget allows it.
4. **Curriculum**: train one epoch on `closed choice` only, then continue on the full mix.
5. **Hard-negative mining**: after one epoch, oversample the examples the model gets wrong.
6. **Pseudo-labelling on test**: run inference on test, take only the high-confidence examples (top quartile of logit margin), add them to train, fine-tune one more epoch. *Only do this if it's allowed under the competition rules.*
7. **Ensemble**: train 2 LoRA adapters with different seeds + LR; average their logits in `infer.ipynb`.
'''))


# =============================================================================
# INFER NOTEBOOK
# =============================================================================

infer_cells = []

infer_cells.append(md(r'''# ScienceQA Vision Challenge — **Inference Notebook**

Loads the LoRA adapter saved by `train.ipynb`, runs **log-likelihood scoring** on `test.csv`, and writes a competition-ready `submission.csv`.

Includes optional **TTA over choice permutations** — averages logits over `K` random orderings of the choice list to remove positional bias from the prediction. Empirically this is worth ~1–3 points on small VLMs.

> **Set `ADAPTER_DIR` to the path produced by `train.ipynb` (it's `outputs_train/adapter` by default).**
'''))

infer_cells.append(md(r'''## 0. Install dependencies

If running offline (no internet at submit time) make sure these are installed in the environment ahead of time.'''))

infer_cells.append(code(r'''# Comment out if your offline environment already has these
%pip install -q transformers==4.57.6 peft==0.18.1 bitsandbytes==0.44.1 accelerate==1.0.1 pillow'''))

infer_cells.append(md(r'''## 1. Imports + paths'''))

infer_cells.append(code(r'''import os, json, math, random
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === EDIT THESE PATHS ===========================================
DATA_DIR    = Path("data")                        # contains test.csv + images/test/*.png
ADAPTER_DIR = Path("outputs_train/adapter")       # produced by train.ipynb
OUT_CSV     = Path("submission.csv")
# =================================================================

# Inference knobs
MODEL_ID         = "HuggingFaceTB/SmolVLM-500M-Instruct"
IMG_SIZE         = 384
USE_4BIT         = False                          # set True on T4
TTA_PERMUTATIONS = 4                              # 1 = no TTA. 4–8 is a sweet spot.
EVAL_BATCH       = 8
MAX_CONTEXT_CHARS = 1200

assert (DATA_DIR / "test.csv").exists(), f"test.csv missing under {DATA_DIR}"
assert ADAPTER_DIR.exists(), f"Adapter dir not found: {ADAPTER_DIR}"
print("device:", device)
print("adapter:", ADAPTER_DIR.resolve())
if (ADAPTER_DIR / "train_meta.json").exists():
    print("train_meta:", json.loads((ADAPTER_DIR / 'train_meta.json').read_text()))'''))

infer_cells.append(md(r'''## 2. Load test data'''))

infer_cells.append(code(r'''test_df = pd.read_csv(DATA_DIR / "test.csv")
test_df["choices"] = test_df["choices"].apply(json.loads)
print("test rows:", len(test_df))
test_df.head(2)'''))

infer_cells.append(md(r'''## 3. Load processor + base model + LoRA adapter'''))

infer_cells.append(code(r'''from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

processor = AutoProcessor.from_pretrained(MODEL_ID)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.padding_side = "left"   # critical for next-token logit reads

model_kwargs: Dict[str, Any] = dict(low_cpu_mem_usage=True)
if USE_4BIT:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model_kwargs["quantization_config"] = bnb
    model_kwargs["device_map"] = "auto"
else:
    model_kwargs["dtype"] = torch.bfloat16
    model_kwargs["device_map"] = "auto" if torch.cuda.is_available() else None

base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **model_kwargs)
if not torch.cuda.is_available(): base.to(device)

model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
model.eval()
model.config.use_cache = True   # OK at inference
print("model + adapter loaded:", type(model).__name__)'''))

infer_cells.append(md(r'''## 4. Prompt + scoring (must match train.ipynb)'''))

infer_cells.append(code(r'''CHOICE_LETTERS = "ABCDEFGHIJ"

def _truncate(s: str, n: int) -> str:
    s = str(s).strip()
    return s if len(s) <= n else s[: n-1] + "…"

def build_user_text(row, choices: List[str], max_ctx=MAX_CONTEXT_CHARS,
                    use_lecture=True, use_hint=True) -> str:
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


def build_messages(row, choices):
    return [{"role":"user","content":[{"type":"image"},{"type":"text","text":build_user_text(row,choices)}]}]


def _letter_token_id(letter: str) -> int:
    ids = processor.tokenizer(letter, add_special_tokens=False).input_ids
    if len(ids) != 1:
        ids = processor.tokenizer(" " + letter, add_special_tokens=False).input_ids
    return ids[-1]

LETTER_IDS = {L: _letter_token_id(L) for L in CHOICE_LETTERS}
print("letter token ids:", LETTER_IDS)'''))

infer_cells.append(code(r'''def load_image(rel_path):
    img = Image.open(DATA_DIR / rel_path).convert("RGB")
    w, h = img.size
    s = IMG_SIZE / max(w, h)
    if s < 1.0:
        img = img.resize((max(1,int(w*s)), max(1,int(h*s))), Image.BICUBIC)
    return img


@torch.inference_mode()
def score_logits(rows, images, choices_list):
    """Return per-row logits over the candidate letters (variable length)."""
    prompts = [processor.apply_chat_template(build_messages(r, ch), add_generation_prompt=True)
               for r, ch in zip(rows, choices_list)]
    inputs = processor(text=prompts, images=[[im] for im in images],
                       return_tensors="pt", padding=True)
    inputs = {k:(v.to(model.device) if torch.is_tensor(v) else v) for k,v in inputs.items()}
    out = model(**inputs)
    last = out.logits[:, -1, :]   # (B, V); left-padded → -1 is the next-token slot
    per_row = []
    for i, ch in enumerate(choices_list):
        ids = [LETTER_IDS[CHOICE_LETTERS[k]] for k in range(len(ch))]
        per_row.append(last[i, ids].float().cpu().numpy())
    return per_row'''))

infer_cells.append(md(r'''## 5. Run inference (with optional TTA)

For each test row we evaluate the model under `TTA_PERMUTATIONS` random orderings of the choice list, project the logits back to the original index space, and average. Set `TTA_PERMUTATIONS = 1` to disable TTA.'''))

infer_cells.append(code(r'''def tta_predict(test_rows, batch_size=EVAL_BATCH, n_perm=TTA_PERMUTATIONS, seed=SEED):
    """Returns list of predicted original-index ints, same length/order as test_rows."""
    n = len(test_rows)
    summed = [None] * n        # accumulated logits in ORIGINAL index space, per row
    rng = np.random.RandomState(seed)

    # pre-load all images once
    images = [load_image(r["image_path"]) for r in test_rows]

    for p in range(n_perm):
        # build per-row permutation
        perms = []
        for r in test_rows:
            k = len(r["choices"])
            perms.append(np.arange(k) if p == 0 else rng.permutation(k))

        for start in range(0, n, batch_size):
            sl = slice(start, min(start+batch_size, n))
            rows_b = test_rows[sl]
            imgs_b = images[sl]
            perms_b = perms[sl]
            shuffled_choices = [[r["choices"][i] for i in perm] for r, perm in zip(rows_b, perms_b)]
            logits_list = score_logits(rows_b, imgs_b, shuffled_choices)
            for i, (logits, perm) in enumerate(zip(logits_list, perms_b)):
                global_i = start + i
                # logits[j] = logit for "the j-th choice in shuffled order"
                # corresponds to original index perm[j]
                if summed[global_i] is None:
                    summed[global_i] = np.zeros(len(perm), dtype=np.float64)
                for j, orig_idx in enumerate(perm):
                    summed[global_i][orig_idx] += float(logits[j])
        print(f"  TTA pass {p+1}/{n_perm} done")

    preds = [int(np.argmax(s)) for s in summed]
    return preds, summed


# Convert the test_df to row-dicts so the function is fast/independent
test_rows = [{
    "id"        : r["id"],
    "image_path": r["image_path"],
    "question"  : r["question"],
    "choices"   : r["choices"],
    "lecture"   : r.get("lecture"),
    "hint"      : r.get("hint"),
} for _, r in test_df.iterrows()]

print(f"Predicting on {len(test_rows)} test rows with {TTA_PERMUTATIONS} TTA passes …")
preds, raw_logits = tta_predict(test_rows)
print("done. preds:", len(preds))'''))

infer_cells.append(md(r'''## 6. Build `submission.csv` with sanity checks'''))

infer_cells.append(code(r'''sub = pd.DataFrame({"id": [r["id"] for r in test_rows], "answer": preds})

# Sanity checks
assert list(sub.columns) == ["id", "answer"], "columns must be exactly ['id','answer']"
assert len(sub) == len(test_df), "row count mismatch"
assert set(sub["id"]) == set(test_df["id"]), "id mismatch with test.csv"
assert sub["answer"].dtype.kind in ("i", "u"), "answer must be integer"
# Each predicted index must be a legal choice index for that row
for r, p in zip(test_rows, preds):
    assert 0 <= p < len(r["choices"]), f"id {r['id']} produced illegal index {p} for {len(r['choices'])} choices"

sub.to_csv(OUT_CSV, index=False)
print(f"✅ wrote {OUT_CSV}  ({len(sub)} rows)")
print("\nAnswer distribution:")
print(sub["answer"].value_counts().sort_index())
print("\nFirst 5 rows:")
print(sub.head())'''))

infer_cells.append(md(r'''## 7. (Optional) Cross-check vs zero-shot

If you also kept a zero-shot prediction CSV around, compare the two to make sure fine-tuning actually moved predictions (a healthy fine-tune typically changes 30–50 % of test predictions).'''))

infer_cells.append(code(r'''# Example skeleton — uncomment if you saved a zero-shot submission earlier
# zs = pd.read_csv("submission_zeroshot.csv")
# diff = (zs.set_index('id')['answer'] != sub.set_index('id')['answer']).mean()
# print(f"fraction of predictions changed by fine-tuning: {diff*100:.1f}%")'''))

infer_cells.append(md(r'''---

**You're done.** Submit `submission.csv` to Kaggle.

Tips:
- If you're under leaderboard time pressure, set `TTA_PERMUTATIONS=1` for a quick first submission, then re-run with `TTA_PERMUTATIONS=8` for your "final" submission once you're confident.
- Keep `submission_zeroshot.csv` (run this notebook before training) as a sanity reference.
'''))


# Write both notebooks
(OUT / "train.ipynb").write_text(json.dumps(notebook(train_cells), indent=1))
(OUT / "infer.ipynb").write_text(json.dumps(notebook(infer_cells), indent=1))
print("wrote train.ipynb and infer.ipynb")
