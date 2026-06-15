# multilqa

Multilingual health question-answering pipeline fine-tuning mT5-large (or NLLB-200)
on health QA pairs across Akan, Amharic, Luganda, Swahili, and English.

## Project layout

```
multilqa/
├── config.yaml          # central config — all knobs live here
├── data/                # Train.csv, Val.csv, Test.csv
├── scripts/             # runnable entrypoints
│   ├── train.py         # fine-tune the model
│   ├── tune.py          # random-search hyperparameter tuning
│   ├── predict.py       # generate submission.csv from Test.csv
│   ├── evaluate.py      # ROUGE scoring against Val.csv
│   ├── ensemble.py      # average top-k checkpoints by eval metric
│   ├── eda.py           # exploratory data analysis
│   ├── run_all.sh       # quick pipeline: EDA → train → predict → evaluate
│   └── run_tuned.sh     # full pipeline: EDA → tune → train → ensemble → predict → evaluate
├── src/                 # importable library (no runnable entry points)
│   ├── config.py        # load_config()
│   ├── dataset.py       # HealthQADataset, get_lang_label, load_tokenizer
│   ├── metrics.py       # ROUGE scoring utilities
│   ├── modeling.py      # load_model, build_training_args, get_model_type
│   ├── trainer.py       # LanguageBalancedSeq2SeqTrainer, build_language_weights
│   └── retrieval.py     # build_retrieval_map, normalize_question
├── slurm/               # SLURM job launchers
└── docs/TUTORIAL.md     # architecture and methodology guide
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

All scripts are run from the **project root** (where `config.yaml` lives).

## Recommended workflow

### Option A — quick run (use config defaults)

```bash
bash scripts/run_all.sh          # EDA → train → predict → evaluate
bash scripts/run_all.sh --debug  # same with small data subset
```

### Option B — full competition run (tune first, then ensemble)

```bash
bash scripts/run_tuned.sh        # EDA → tune → train → ensemble → predict → evaluate
bash scripts/run_tuned.sh --debug
```

### Step by step

```bash
# 1. Explore the data
python scripts/eda.py

# 2. (Optional) Find best hyperparams — results saved to output/tuning/results.json
python scripts/tune.py
python scripts/tune.py --debug   # fast version on small data

# 3. Train (apply best tuning params automatically if step 2 was run)
python scripts/train.py --from-tuning-results output/tuning/results.json
python scripts/train.py          # or just use config.yaml defaults
python scripts/train.py --debug  # fast smoke-test

# 4. Average top-k checkpoints for a free ROUGE boost
python scripts/ensemble.py

# 5. Generate predictions
python scripts/predict.py                              # uses final_model_dir
python scripts/predict.py --model-dir output/ensemble_model  # uses ensemble

# 6. Score against validation set
python scripts/evaluate.py output/submission.csv
```

## Switching models

Edit `config.yaml`:

```yaml
model:
  name: google/mt5-large          # default
  # name: facebook/nllb-200-1.3B  # NLLB alternative for low-resource languages
```

Model type (mt5 / nllb) is auto-detected from the name. No other changes needed.

## SLURM cluster

```bash
sbatch slurm/train.slurm
sbatch slurm/tune.slurm
bash   slurm/pipeline.slurm      # submits all jobs with dependency chain
```

## Notes

- `--debug` limits training to ~500 train / 100 val rows (stratified by language).
- SLURM scripts expect a virtual environment at `venv/`.
- See `docs/TUTORIAL.md` for a full architecture and methodology guide.
