# MultiLQA — Architecture, Concepts, and Methodology Guide

This document explains the core ideas behind every component of the pipeline: *why* each technique was chosen, *what* it does mathematically, and *how* it manifests in the code.

---

## Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [Model Architecture — mT5](#2-model-architecture--mt5)
3. [Alternative Model — NLLB-200](#3-alternative-model--nllb-200)
4. [Tokenisation and the SentencePiece Vocabulary](#4-tokenisation-and-the-sentencepiece-vocabulary)
5. [Language Prefix Prompting vs. Language Token Conditioning](#5-language-prefix-prompting-vs-language-token-conditioning)
6. [Sequence-to-Sequence Training](#6-sequence-to-sequence-training)
7. [The Encoder–Decoder Loss Function](#7-the-encoderdecoder-loss-function)
8. [Language-Balanced Sampling](#8-language-balanced-sampling)
9. [Optimisation — AdaFactor](#9-optimisation--adafactor)
10. [Mixed Precision — BF16](#10-mixed-precision--bf16)
11. [Gradient Accumulation and Gradient Checkpointing](#11-gradient-accumulation-and-gradient-checkpointing)
12. [Dynamic Padding and DataCollatorForSeq2Seq](#12-dynamic-padding-and-datacollatorforseq2seq)
13. [Evaluation — ROUGE Metrics](#13-evaluation--rouge-metrics)
14. [LLM-as-a-Judge](#14-llm-as-a-judge)
15. [Inference — Beam Search](#15-inference--beam-search)
16. [Retrieval Fallback](#16-retrieval-fallback)
17. [Hyperparameter Tuning — Random Search](#17-hyperparameter-tuning--random-search)
18. [Checkpoint Ensembling — Weight Averaging](#18-checkpoint-ensembling--weight-averaging)
19. [Code Organisation — `src/` and `scripts/`](#19-code-organisation--src-and-scripts)
20. [Pipeline Flow End-to-End](#20-pipeline-flow-end-to-end)
21. [Key Configuration Knobs](#21-key-configuration-knobs)

---

## 1. Problem Framing

### Task type: Conditional text generation

The task is **open-ended question answering**: given a health question in one of five languages (Akan, Amharic, Luganda, Swahili, English), generate a fluent, factually correct answer in the *same* language.

This is a **sequence-to-sequence (seq2seq)** problem. The input sequence is the question; the output sequence is the answer. Unlike classification, there is no fixed label set — the model must generate an arbitrary string token by token.

### Why not classification or extractive QA?

- **Classification** maps an input to one of N fixed categories. Answers here are free-form text — there is no bounded label set.
- **Extractive QA** (like BERT-style span extraction) copies a span directly from a source document. This task has no source document to extract from — the model must draw on parametric knowledge learned during pre-training and fine-tuning.
- **Generative QA** (seq2seq) can synthesise answers that are not verbatim copies, which is necessary for fluent, contextually appropriate health guidance.

### Why multilingual, and why is it hard?

Standard language models are trained predominantly on English web text. Languages like Akan and Luganda have orders of magnitude less training data on the internet. A model fine-tuned only on English will:

- Fail to understand questions posed in Akan script.
- Generate English responses to non-English questions.
- Produce answers that are factually correct but linguistically inappropriate.

The solution is to start from a model that was pre-trained on many languages simultaneously, then fine-tune on task-specific multilingual data.

---

## 2. Model Architecture — mT5

### What mT5 is

`mT5` (Multilingual T5) is an encoder–decoder transformer pre-trained by Google on the **mC4** corpus — a cleaned multilingual version of Common Crawl covering 101 languages. The default variant used here, `google/mt5-large`, has approximately 1.2 billion parameters.

### The transformer encoder

The encoder reads the full input sequence in parallel and produces a sequence of **contextualised hidden states** — one vector per input token. "Contextualised" means each token's representation is influenced by all other tokens via **self-attention**.

Self-attention computes, for each token, a weighted sum over all other tokens' values:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

where Q (queries), K (keys), and V (values) are linear projections of the token embeddings. The scaling factor `sqrt(d_k)` prevents the dot products from growing too large, which would push the softmax into saturated (near-zero gradient) regions.

### The transformer decoder

The decoder generates the output sequence **one token at a time** in an autoregressive manner. At each step it attends to:

1. **Its own previously generated tokens** (masked self-attention — future tokens are masked so the model cannot "cheat").
2. **The encoder's output** (cross-attention — the decoder asks "which parts of the input are relevant right now?").

This cross-attention is the mechanism by which the model conditions its answer generation on the input question.

### Why encoder–decoder instead of decoder-only?

Decoder-only models (GPT-style) generate text left-to-right but have no separate encoding of the input. Encoder–decoder models give the decoder a richly compressed representation of the entire input before it begins generating. For tasks where the input (question) and output (answer) are structurally different and of similar length, this separation typically performs better.

### `tie_word_embeddings = False`

By default, many T5 variants share the embedding matrix between the encoder input, decoder input, and the output projection (lm head). mT5 checkpoints are distributed with **untied weights** — the encoder and decoder have separate embedding matrices. Forcing `tie_word_embeddings = True` would corrupt the loaded weights. The code explicitly sets:

```python
model.config.tie_word_embeddings = False
```

---

## 3. Alternative Model — NLLB-200

### What NLLB-200 is

`facebook/nllb-200-1.3B` (No Language Left Behind, 200 languages, 1.3B parameters) is Meta's encoder–decoder transformer specifically designed for low-resource translation and multilingual generation. Unlike mT5, which was pre-trained with a general language modelling objective, NLLB-200 was pre-trained explicitly on parallel corpora across 200 languages — including all five competition languages — with an emphasis on African and other under-resourced languages.

### Why NLLB may outperform mT5 on this task

| Property | mT5-large | NLLB-200-1.3B |
|---|---|---|
| Pre-training objective | Masked span prediction (T5) | Seq2seq translation (200 langs) |
| Language coverage | 101 languages (mostly web text) | 200 languages (parallel corpora) |
| Low-resource African lang focus | Moderate | Strong (explicit training goal) |
| Conditioning mechanism | Text prefix ("akan question: ...") | Dedicated language token (forced_bos) |

### How NLLB conditions on language

NLLB does not use text prefixes. Instead, it uses **forced beginning-of-sequence (BOS) tokens** — each language has a unique token in the vocabulary (e.g. `aka_Latn`, `amh_Ethi`, `swh_Latn`). The tokeniser's `src_lang` and `tgt_lang` attributes set the context, and at generation time:

```python
forced_bos_id = tokenizer.convert_tokens_to_ids("swh_Latn")
model.generate(..., forced_bos_token_id=forced_bos_id)
```

This forces the very first generated token to be the language marker, which strongly conditions all subsequent output to remain in the target language.

### Switching between mT5 and NLLB-200

Change one line in `config.yaml`:

```yaml
model:
  name: facebook/nllb-200-1.3B   # or facebook/nllb-200-distilled-600M for faster runs
```

All downstream code (dataset construction, tokenisation, generation) detects the model type automatically via `src/modeling.py:get_model_type()`.

---

## 4. Tokenisation and the SentencePiece Vocabulary

### SentencePiece

mT5 uses a **SentencePiece** tokeniser with a **250,000-token unigram vocabulary** trained across all 101 mC4 languages. This is much larger than the 30–50k vocabularies typical for English-only models, because it needs to cover character sequences across dozens of scripts (Latin, Ethiopic for Amharic, etc.).

NLLB-200 uses a similar SentencePiece vocabulary of ~256k tokens trained on its 200-language corpus, plus one dedicated language-code token per language.

### Subword tokenisation

SentencePiece performs **subword segmentation**: frequent whole words become single tokens; rare words are split into smaller pieces. For example, an Akan word that occurs rarely might be split into two or three subword pieces. This:

- Keeps the vocabulary finite regardless of language vocabulary size.
- Handles out-of-vocabulary words gracefully by decomposing them.
- Allows the model to share representations across related words (e.g. verb stems and their inflections).

### Token budget

- `input_max_len = 256`: questions are truncated or padded to at most 256 tokens.
- `target_max_len = 512`: reference answers during training can be up to 512 tokens.
- Inference `max_new_tokens = 384`: the model generates at most 384 tokens per answer.

These limits matter because transformer self-attention is O(n²) in sequence length — doubling the length quadruples the compute.

---

## 5. Language Prefix Prompting vs. Language Token Conditioning

### mT5: Text prefix prompting

The mT5 model cannot know which language to generate in unless it is told. The technique used here is to prepend a natural-language prefix to every input:

```
"akan question: <question text>"
"amharic question: <question text>"
```

This is the standard **task prefix** pattern from the original T5 paper ("translate English to German: ..."). The model learns during fine-tuning to associate these prefixes with the corresponding output language and domain.

### NLLB: Language token conditioning

For NLLB-200, **no text prefix is added**. Instead, the tokeniser receives per-example `src_lang` and `tgt_lang` codes, which embed as language vectors in the model's internal representation. At generation time, `forced_bos_token_id` ensures the decoder starts in the correct language.

Because the NLLB approach requires a uniform `forced_bos_token_id` per model call (the token must be the same for all items in a batch), `predict.py` groups non-retrieved test examples by language before calling `model.generate`, then rejoins results in the original order.

### How the language label is determined

`src/dataset.py:get_lang_label()` reads the `subset` column (e.g. `"Aka_Gha"`, `"Swa_Tz"`), extracts the three-letter language code, and maps it via `config.yaml:language_map`:

```
Aka → akan,  Amh → amharic,  Lug → luganda,  Swa → swahili,  Eng → english
```

If the `subset` column is absent or empty, `langdetect` is used as a fallback to detect the language from the question text itself.

---

## 6. Sequence-to-Sequence Training

### Teacher forcing

During training, the decoder does not use its own previous predictions. Instead, it receives the **ground-truth previous token** at each step (called *teacher forcing*). This makes training stable — errors do not compound — but creates a train/inference discrepancy (exposure bias), which is an accepted trade-off.

### Label masking (`-100`)

The reference answer is tokenised and padded to `target_max_len`. Padding positions are set to `-100`. PyTorch's cross-entropy loss ignores indices equal to `-100`, so padding does not contribute to the loss. This is handled in `src/dataset.py`:

```python
labels[labels == self.pad_id] = -100
```

When `DataCollatorForSeq2Seq` adds further padding to equalise lengths within a batch, it also uses `-100` for the added positions.

---

## 7. The Encoder–Decoder Loss Function

### Cross-entropy loss

The training objective is token-level **cross-entropy** between the model's predicted probability distribution over the vocabulary and the one-hot ground-truth token:

```
L = -sum_t log P(y_t | y_{<t}, x)
```

where `x` is the input sequence, `y_t` is the ground-truth token at position `t`, and the sum runs over all non-padding positions.

Minimising this loss is equivalent to maximising the likelihood of the ground-truth answer token by token.

### Why label smoothing is disabled

mT5 uses a shared softmax layer that can be numerically unstable with label smoothing when `tie_word_embeddings = False`. The config enforces:

```yaml
label_smoothing_factor: 0.0
```

This is a known mT5 constraint — label smoothing distributes probability mass away from the true token, and in mT5's architecture this interacts poorly with how the decoder input IDs are computed.

---

## 8. Language-Balanced Sampling

### The imbalance problem

The training set contains ~29,815 records split across nine language-country configurations. English examples typically outnumber Akan or Luganda examples by a large factor. An unweighted training loop sees far more English data per epoch, biasing the model toward English patterns. This directly harms performance on low-resource language questions — exactly the languages the competition prioritises.

### Inverse-frequency weighting

`src/trainer.py:build_language_weights()` computes a weight for each training example:

```
weight(i) = 1 / count(lang(i))^alpha
```

where `alpha` controls the strength of balancing (1.0 = full inverse-frequency; 0.0 = no balancing). These weights are normalised to have mean 1.0 so the effective learning rate scale is unchanged.

### WeightedRandomSampler

PyTorch's `WeightedRandomSampler` draws training examples *with replacement* according to these weights. Low-resource language examples are sampled more frequently; high-resource examples less frequently. The total number of samples per epoch equals the dataset size.

This is implemented in `src/trainer.py:LanguageBalancedSeq2SeqTrainer`, a subclass of HuggingFace's `Seq2SeqTrainer` that overrides `get_train_dataloader` to swap in the weighted sampler.

---

## 9. Optimisation — AdaFactor

### Why not Adam?

Adam maintains per-parameter first and second moment estimates, requiring **2× the model parameter count** in optimiser state memory. For a 1.2B parameter model, Adam's state alone occupies tens of gigabytes.

**AdaFactor** approximates the second moment matrix as a factored outer product of row and column statistics, reducing optimiser memory to roughly **O(n + m)** per matrix instead of **O(n × m)**. This makes it the standard optimiser for T5-family models.

### Learning rate schedule

A **warmup phase** linearly increases the learning rate from 0 to the target over the first `warmup_ratio` fraction of training steps. After warmup, AdaFactor uses its internal adaptive scaling. Warmup prevents large, destabilising gradient updates in the first steps when the model weights are far from their trained state.

```yaml
learning_rate: 5.0e-4
warmup_ratio: 0.03
```

---

## 10. Mixed Precision — BF16

### Why not FP16?

Standard 16-bit floating point (FP16) has a dynamic range of approximately 6×10⁻⁵ to 6.5×10⁴. mT5's logits and gradients can exceed this range, causing **overflow to infinity** or **underflow to zero** — training diverges.

**BF16** (Brain Float 16) uses the same 8 exponent bits as FP32 (dynamic range 10⁻³⁸ to 3.4×10³⁸) but only 7 mantissa bits instead of 23. It sacrifices precision for range. For deep learning, range matters more than precision — gradients span many orders of magnitude, but don't need to be known to 7 decimal places.

```yaml
fp16: false   # would cause mT5 overflow
bf16: true    # stable on Ampere/Ada GPUs (e.g. L40S)
```

BF16 is only enabled when a CUDA GPU is available, since BF16 arithmetic is hardware-accelerated on Ampere and later NVIDIA architectures.

---

## 11. Gradient Accumulation and Gradient Checkpointing

### Gradient accumulation

With limited GPU memory, you cannot fit a large batch in a single forward pass. Gradient accumulation runs `gradient_accumulation_steps` smaller forward passes and **sums the gradients** before applying a single optimiser step. The result is mathematically equivalent to training with a larger effective batch:

```
effective_batch = per_device_train_batch_size × gradient_accumulation_steps
               = 8 × 4 = 32
```

Larger effective batches produce more stable gradient estimates and often allow a higher learning rate.

### Gradient checkpointing

Normally, the forward pass caches all intermediate activations for use during the backward pass. For a large model with long sequences, these cached activations dominate GPU memory.

**Gradient checkpointing** discards intermediate activations during the forward pass and **recomputes them on-the-fly** during the backward pass. This trades extra computation (roughly 30–40% more FLOPs) for significantly reduced memory. Enabled via:

```yaml
gradient_checkpointing: true
```

---

## 12. Dynamic Padding and DataCollatorForSeq2Seq

### The problem with static padding

Without a collator, every example in the dataset is padded to the maximum possible length (`input_max_len = 256`, `target_max_len = 512`) regardless of the actual question or answer length. Most health questions are far shorter than 256 tokens. The GPU processes padding tokens at full cost — this is wasted compute.

### Dynamic (per-batch) padding

`DataCollatorForSeq2Seq` receives a batch of variable-length examples and pads each sequence in the batch to **the length of the longest sequence in that batch**. A batch of short questions wastes almost no compute; a batch containing a long question pads all others to match it.

`pad_to_multiple_of=8` ensures padded lengths are multiples of 8, which aligns data in memory for efficient GPU tensor cores.

```python
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)
```

The `label_pad_token_id=-100` argument ensures any padding the collator adds to label sequences is immediately masked out from the loss, consistent with the dataset's own masking.

---

## 13. Evaluation — ROUGE Metrics

### ROUGE-1

ROUGE-1 (Recall-Oriented Understudy for Gisting Evaluation) measures **unigram overlap** between the predicted answer and the reference answer:

```
Precision = |pred_unigrams ∩ ref_unigrams| / |pred_unigrams|
Recall    = |pred_unigrams ∩ ref_unigrams| / |ref_unigrams|
F1        = 2 × Precision × Recall / (Precision + Recall)
```

ROUGE-1 F1 captures keyword overlap. A prediction that contains the right medical terms scores well even if the phrasing differs.

### ROUGE-L

ROUGE-L uses the **Longest Common Subsequence (LCS)** rather than unigram overlap:

```
LCS_precision = LCS(pred, ref) / len(pred)
LCS_recall    = LCS(pred, ref) / len(ref)
LCS_F1        = 2 × LCS_p × LCS_r / (LCS_p + LCS_r)
```

ROUGE-L rewards predictions that preserve the order of content words, capturing sentence-level structure and fluency better than ROUGE-1.

### Per-language breakdown

`scripts/train.py` runs a separate ROUGE evaluation for each language after training. This diagnoses whether the model is performing well uniformly or is strong in English but weak in Akan — which would not be visible from the aggregate score alone.

### Leaderboard score

The final leaderboard score is a weighted mean:

```
score = 0.37 × ROUGE-1 F1 + 0.37 × ROUGE-L F1 + 0.26 × LLM-judge
```

### Important: training eval vs. inference token limits

During training, `generation_max_length = 128` is used for checkpoint evaluation. At inference, `max_new_tokens = 384` is used. These must be in the same ballpark — a large gap means the model is selected based on truncated-output metrics that don't reflect submission quality.

---

## 14. LLM-as-a-Judge

The third evaluation metric (26% of the score) has an LLM read both the reference answer and the model's prediction, then score the prediction on:

- **Factual accuracy** — Is the health information correct?
- **Completeness** — Does the answer address all parts of the question?
- **Language appropriateness** — Is the response in the correct language and culturally appropriate?

The raw score (1–5) is normalised to [0, 1]. This metric catches failure modes that ROUGE cannot: a prediction that is in entirely the wrong language can still have high ROUGE overlap with a same-language reference if individual tokens happen to match.

The LLM proxy in `scripts/evaluate.py` uses `(ROUGE-1 + ROUGE-L) / 2` as a local approximation, but this underestimates the importance of generating in the correct language.

---

## 15. Inference — Beam Search

### Greedy decoding vs. beam search

**Greedy decoding** picks the single highest-probability token at each step. It is fast but locally optimal — a high-probability first token might lead into a low-probability continuation.

**Beam search** maintains `num_beams` candidate sequences simultaneously. At each step, each beam is extended by all vocabulary tokens, producing `num_beams × vocab_size` candidates. The top `num_beams` by cumulative log-probability are kept. After all steps, the highest-scoring complete sequence is returned.

```yaml
num_beams: 8
```

Beam search finds higher-probability sequences than greedy decoding, typically producing more fluent and complete answers.

### No-repeat n-gram constraint

```yaml
no_repeat_ngram_size: 3
```

The model is forbidden from repeating any 3-gram that has already appeared in the generated output. This prevents degenerate repetition loops that sometimes appear in seq2seq models (e.g. "Take medication. Take medication. Take medication...").

### Length penalty

```yaml
length_penalty: 1.0
```

Beam search normalises scores by sequence length raised to `length_penalty`. Values > 1 favour longer sequences; values < 1 favour shorter ones. 1.0 applies no length bias, so the raw log-probability is used.

### Early stopping

```yaml
early_stopping: true
```

With `early_stopping=True`, beam search stops as soon as all beams have generated an end-of-sequence token, rather than running to `max_new_tokens`. This speeds up inference on short answers without sacrificing quality on long ones.

---

## 16. Retrieval Fallback

Some test questions are identical to training questions. For these, the model-generated answer is unnecessary — we can return the known ground-truth answer directly and guarantee high ROUGE overlap.

`src/retrieval.py:build_retrieval_map()` builds a dictionary mapping every training question to its answer. Questions are **normalised** before keying:

```python
def normalize_question(text: str) -> str:
    text = str(text).lower().strip()
    return re.sub(r"\s+", " ", text)
```

This collapses whitespace and lowercases, so minor formatting differences between train and test (extra spaces, case) do not prevent a match.

Before calling the model, each test question is looked up in this map. Exact normalised-matches are returned immediately, bypassing generation entirely. This is a deterministic improvement for any overlap between train and test question sets.

---

## 17. Hyperparameter Tuning — Random Search

### Why random search?

Grid search evaluates all combinations of a discrete hyperparameter grid. With 4 learning rates × 2 weight decays × 3 warmup ratios × ... the number of combinations grows exponentially (the *curse of dimensionality*). Most combinations waste compute in poor regions.

**Random search** samples hyperparameter combinations uniformly at random from each dimension. Empirically, random search finds equally good or better configurations than grid search with far fewer trials, because hyperparameter landscapes are often low-dimensional — only a few parameters matter significantly, and random search covers those important dimensions well even with few trials.

### Trial protocol

Each trial runs `scripts/train.py` as a subprocess with 2 epochs on the full dataset, saving no model weights (`--skip-save-model`). The trial's ROUGE-L score on the validation set is recorded. After all trials, results are sorted by ROUGE-L and written to `output/tuning/results.json`. The full train then reads this file via `--from-tuning-results`:

```bash
python scripts/train.py --from-tuning-results output/tuning/results.json
```

Inside `train.py`, a three-tier resolution function applies parameters in priority order: CLI flag > tuning result > `config.yaml` default.

### Search space

| Hyperparameter | Options |
|---|---|
| Learning rate | 3e-4, 4e-4, 5e-4, 6e-4 |
| Weight decay | 0.0, 0.01 |
| Warmup ratio | 0.03, 0.06, 0.10 |
| Generation max length | 128, 192 |
| Gradient accumulation steps | 2, 4 |
| Balanced sampling | true, false |
| Balance alpha | 0.7, 1.0, 1.3 |

`label_smoothing_factor` is forced to 0.0 for mT5 regardless of what is sampled.

---

## 18. Checkpoint Ensembling — Weight Averaging

### The idea

After training, the Trainer saves a checkpoint at each epoch (up to `save_total_limit = 3`). Each checkpoint represents the model at a different point in the loss landscape. **Weight averaging** arithmetically averages the parameter tensors of multiple checkpoints:

```
W_ensemble = (W_ckpt1 + W_ckpt2 + ... + W_ckptK) / K
```

This is not the same as averaging predictions — it is an average in weight space. Empirically, weight averaging over the last few checkpoints often achieves better generalisation than any single checkpoint, because the averaged weights tend to sit in a flatter region of the loss landscape (lower sharpness) that is more robust to small input distribution shifts.

### Why flat minima generalise better

The concept of **loss landscape sharpness** posits that a sharp minimum (steep walls) corresponds to a solution whose performance degrades quickly if the weights shift slightly — i.e. a solution that overfits to the training distribution. A flat minimum is surrounded by a large region of low loss — more forgiving of small perturbations, more likely to generalise.

Weight averaging across multiple checkpoints effectively averages solutions from different points in the optimisation trajectory, which tends to produce a point in a flatter basin.

### Implementation in `scripts/ensemble.py`

The script:
1. Scans `output/checkpoints/` for all `checkpoint-N` directories.
2. Reads each checkpoint's `trainer_state.json` to find the last logged `eval_rougeL` value.
3. Selects the top-K checkpoints by that metric.
4. Loads and averages their state dicts in `float32` (upcasting to prevent precision loss from accumulating integer-truncation errors).
5. Loads the model architecture from the best checkpoint, applies the averaged state dict, and saves the result to `output/ensemble_model/`.

The ensemble model is then used for inference by pointing `predict.py` at it:

```bash
python scripts/predict.py --model-dir output/ensemble_model
```

---

## 19. Code Organisation — `src/` and `scripts/`

The project separates reusable library code from runnable entrypoints.

```
multilqa/
├── config.yaml                   ← single source of truth for all hyperparameters
│
├── src/                          ← importable Python package (library)
│   ├── __init__.py
│   ├── config.py                 ← load_config() — YAML loader
│   ├── dataset.py                ← HealthQADataset, get_lang_label, load_tokenizer
│   ├── metrics.py                ← make_compute_metrics, per_language_eval, build_scorer
│   ├── modeling.py               ← load_model, build_training_args, get_model_type
│   ├── retrieval.py              ← build_retrieval_map, normalize_question
│   └── trainer.py                ← LanguageBalancedSeq2SeqTrainer, build_language_weights
│
├── scripts/                      ← thin runnable entrypoints
│   ├── eda.py                    ← exploratory data analysis
│   ├── tune.py                   ← random hyperparameter search
│   ├── train.py                  ← fine-tuning (--from-tuning-results supported)
│   ├── ensemble.py               ← checkpoint weight averaging
│   ├── predict.py                ← inference + retrieval (--model-dir supported)
│   ├── evaluate.py               ← ROUGE + LLM-proxy scoring
│   ├── run_all.sh                ← simple linear pipeline (no tuning)
│   └── run_tuned.sh              ← full pipeline with tuning + ensembling
│
├── slurm/                        ← SLURM job files for HPC submission
│   ├── train.slurm
│   ├── predict.slurm
│   ├── tune.slurm
│   └── evaluate.slurm
│
├── data/                         ← competition CSVs (Train.csv, Val.csv, Test.csv, ...)
├── output/                       ← generated at runtime
│   ├── checkpoints/              ← epoch checkpoints (checkpoint-N/)
│   ├── final_model/              ← single best checkpoint saved after training
│   ├── ensemble_model/           ← weight-averaged checkpoint from ensemble.py
│   ├── tuning/results.json       ← hyperparameter trial results
│   ├── submission.csv            ← final prediction output
│   ├── train_metrics.json        ← best ROUGE scores + resolved params
│   └── eval_results.json         ← post-submission evaluation
│
└── docs/
    └── TUTORIAL.md               ← this document
```

Every script in `scripts/` inserts the project root onto `sys.path` so that `from src.X import Y` resolves correctly regardless of the current working directory:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

All configuration is centralised in `config.yaml` and loaded by every script via `src/config.py:load_config()`. There is no duplicated YAML loading.

---

## 20. Pipeline Flow End-to-End

### Option A — Simple pipeline (no tuning)

```
data/Train.csv, Val.csv, Test.csv
        │
        ▼
  [1] scripts/eda.py
        Exploratory data analysis: language distribution,
        answer lengths, duplicate detection, null counts.
        → output/eda_report.txt
        → output/answer_length_dist.png
        │
        ▼
  [2] scripts/train.py
        Fine-tunes model (mT5-large or NLLB-200) using config.yaml.
        Saves checkpoint per epoch; loads best by val ROUGE-L.
        Runs per-language ROUGE breakdown.
        → output/checkpoints/checkpoint-N/
        → output/final_model/
        → output/train_metrics.json
        │
        ▼
  [3] scripts/predict.py
        Loads final_model, generates answers for Test.csv.
        Retrieval fallback for normalised-exact-match questions.
        → output/submission.csv
        │
        ▼
  [4] scripts/evaluate.py output/submission.csv
        Scores submission against Val.csv ground truth.
        Per-language breakdown + leaderboard-proxy score.
        → output/eval_results.json
```

Run with: `bash scripts/run_all.sh`

### Option B — Full pipeline with tuning and ensembling

```
data/Train.csv, Val.csv, Test.csv
        │
        ▼
  [1] scripts/eda.py                  (same as Option A)
        │
        ▼
  [2] scripts/tune.py
        Random search over N trials, each 2 epochs.
        → output/tuning/results.json
        │
        ▼
  [3] scripts/train.py
          --from-tuning-results output/tuning/results.json
        Applies best trial params (CLI > tuning > config fallback).
        → output/checkpoints/checkpoint-N/
        → output/final_model/
        → output/train_metrics.json
        │
        ▼
  [4] scripts/ensemble.py
        Selects top-K checkpoints by eval_rougeL.
        Averages their weights in float32.
        → output/ensemble_model/
        │
        ▼
  [5] scripts/predict.py
          --model-dir output/ensemble_model
        Uses the ensemble model instead of final_model.
        → output/submission.csv
        │
        ▼
  [6] scripts/evaluate.py output/submission.csv
        → output/eval_results.json
```

Run with: `bash scripts/run_tuned.sh`

---

## 21. Key Configuration Knobs

All tunable values live in `config.yaml`. Here are the most impactful ones and what to change them for:

| Key | Default | Effect |
|---|---|---|
| `model.name` | `google/mt5-large` | Swap to `facebook/nllb-200-1.3B` for NLLB; `google/mt5-xl` for largest mT5 |
| `training.num_train_epochs` | `5` | More epochs → better fit, diminishing returns after ~5 |
| `training.learning_rate` | `5e-4` | Too high → unstable; too low → slow convergence |
| `training.gradient_accumulation_steps` | `4` | Effective batch = per_device_batch × this; higher = more stable gradients |
| `training.balanced_sampling` | `true` | Enables inverse-frequency language weighting |
| `training.balance_alpha` | `1.0` | 0 = no balancing, 1 = full inverse-frequency, >1 = over-correct |
| `training.generation_max_length` | `128` | Token budget for eval during training; should be close to inference limit |
| `training.save_total_limit` | `3` | Number of checkpoints kept; ensemble.py needs at least 2 |
| `inference.max_new_tokens` | `384` | Max answer length at submission time |
| `inference.num_beams` | `8` | More beams → better quality, slower inference |
| `ensemble.top_k` | `3` | Number of checkpoints to weight-average |
| `ensemble.metric` | `eval_rougeL` | Trainer metric used to rank checkpoints |
| `tuning.trials` | `8` | More trials → better chance of finding optimal hyperparams |
| `model.input_max_len` | `256` | Truncates questions longer than this |
| `model.target_max_len` | `512` | Truncates reference answers longer than this during training |
| `model.nllb_language_map` | (see config) | Maps language labels to NLLB BCP-47 codes; only used when model is NLLB |
