# multilqa Tutorial

## Project overview

`multilqa` is a multilingual health question-answering training pipeline built around Hugging Face Transformers and PyTorch. It trains `google/mt5-base` on a CSV dataset containing multilingual health QA pairs, with support for:

- language-aware prefixing and inference prompts
- balanced sampling by language
- BF16 training for mT5 stability
- hyperparameter tuning via random search
- SLURM-based cluster execution

## Key files and architecture

### `config.yaml`

This is the central configuration file. It defines:

- data file paths and CSV column names
- language mapping rules
- model and tokenizer settings
- training defaults and debug limits
- hyperparameter search space for tuning
- inference and evaluation output settings

### `dataset.py`

Contains data preprocessing and the dataset wrapper.

- `HealthQADataset` tokenizes text and target pairs using `AutoTokenizer`
- `get_lang_label()` resolves the language from the `subset` column
- fallback language detection uses `langdetect`
- input examples are prefixed with language-aware prompts like `"<lang> question: ..."`

### `train.py`

The main training entrypoint.

It supports:

- loading config from `config.yaml`
- CLI override parameters for tuning and fast experiments
- debug mode with a small stratified subset of languages
- language-balanced sampling via weighted sampling
- mT5-safe training using `bf16` and `gradient_checkpointing`
- custom Rouge metric computation using `rouge_scorer`
- stable decode handling via `to_token_ids()`

### `tune.py`

A lightweight hyperparameter random search wrapper.

- samples a search space from `config.yaml`
- runs multiple `train.py` trials sequentially
- supports `--debug` for fast subset tuning
- writes trial metrics and a `results.json` summary
- automatically forces `label_smoothing_factor=0.0` for mT5

### SLURM launcher scripts

- `slurm/train.slurm` executes `train.py` on a GPU node
- `slurm/tune.slurm` executes `tune.py` on a GPU node

Both scripts:

- activate the repo `venv`
- verify CUDA availability
- enable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- write logs to `logs/`

## Data format

The training data is expected as CSV files with columns:

- `ID`
- `input` (question text)
- `output` (answer text)
- `subset` (language/country key, e.g. `Aka_Gha`)

Language-normalized labels are derived from the `subset` prefix and then used to build prompts.

## How to run

### Prepare the environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train on full data

```bash
python train.py
```

### Train in debug mode

```bash
python train.py --debug
```

### Train on cluster via SLURM

```bash
sbatch slurm/train.slurm
```

### Run hyperparameter tuning

```bash
python tune.py
```

### Debug tuner on a small subset

```bash
python tune.py --debug
```

### Run tuning on cluster via SLURM

```bash
sbatch slurm/tune.slurm
```

## Common overrides

The training script supports CLI flags to override configuration values for fast experimentation:

- `--learning-rate`
- `--weight-decay`
- `--warmup-ratio`
- `--label-smoothing-factor`
- `--generation-max-length`
- `--gradient-accumulation-steps`
- `--balanced-sampling` / `--no-balanced-sampling`
- `--balance-alpha`
- `--output-dir`
- `--final-model-dir`
- `--metrics-out`
- `--debug`

## Notes and best practices

- `google/mt5-base` is sensitive to `fp16`; this repo uses `bf16` when CUDA is available.
- Balanced language sampling can improve multilingual generalization, especially for underrepresented languages.
- The debug mode is stratified by language to avoid hiding errors behind single-language samples.
- SLURM scripts assume a local `venv` at `/home/jgibson2/projects/multilqa/venv`. Update the path if your environment differs.

## Where to look next

- `config.yaml` for the current training and tuning defaults
- `train.py` for model training and evaluation behavior
- `tune.py` for hyperparameter search flow
- `slurm/train.slurm` and `slurm/tune.slurm` for cluster execution
