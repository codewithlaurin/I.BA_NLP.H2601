# ART Abductive Commonsense Reasoning

HSLU NLP course project (HS26) comparing frozen FastText embeddings with mean pooling and recurrent encoders on the ART reasoning task. Given two observations and two candidate hypotheses, the model predicts which hypothesis best explains the observations.

[Experiment dashboard: P1-Overview-MeanPooling-and-RNN](https://wandb.ai/laurin-uni-workspace/nlp-classifier?nw=5bi9k4rwebs)

## Dataset

The project uses the course-specific [ART dataset](https://huggingface.co/datasets/amarfurt/art-nlp-hs26), loaded through Hugging Face Datasets.

| Split | Examples | Purpose |
| --- | ---: | --- |
| Train | 16,801 | Optimize model parameters |
| Validation | 1,000 | Compare configurations, select checkpoints, and stop training |
| Test | 1,532 | Evaluate selected checkpoints |

Each example contains `observation_1`, `observation_2`, `hypothesis_1`, `hypothesis_2`, and a label identifying the correct hypothesis. Labels 1 and 2 are mapped to classes 0 and 1. Training entry points load train/validation only unless `--evaluate-test` is supplied.

Preprocessing lowercases text, removes Unicode punctuation, deletes internal apostrophes, normalizes whitespace, and splits on whitespace. There is no stemming, lemmatization, stopword removal, or truncation. A new vocabulary is built from training text, with `<PAD>` and `<UNK>` tokens. FastText supplies subword vectors for training words outside its pretrained dictionary; validation/test words outside the project vocabulary map to `<UNK>`. The padding vector is zero, and the unknown vector is the mean of the ordinary vocabulary vectors. Batches preserve all four fields per example, pad to the longest field in the batch, and retain the final partial batch.

## Models

Both architectures use frozen, 300-dimensional English FastText embeddings from `cc.en.300.bin` and a two-layer classifier:

```text
Features -> Linear -> ReLU -> Dropout -> Linear -> 2 logits
```

| Encoder | Classifier input |
| --- | --- |
| Mean pooling | Masked mean embeddings for each of the four fields, concatenated into 1,200 features |
| Two-layer GRU or LSTM | Top-layer final hidden state for each field, concatenated into `4H` features |
| Two-layer BiGRU or BiLSTM | Final forward and backward hidden states for each field, concatenated into `8H` features |

`H` is recurrent hidden width per direction. The same RNN processes all four fields independently, starting from zero hidden states. Packed sequences exclude padding; LSTM representations use the hidden state rather than the cell state. Relationships between fields are learned by the classifier.

`--hidden-dim` controls classifier width; `--rnn-hidden-size` controls recurrent width. Cross-entropy receives raw logits. Softmax is used only when requesting probabilities.

## Setup

Requirements: Python 3.14 or newer and [uv](https://docs.astral.sh/uv/). CPU, CUDA, and Apple MPS are supported; `--device auto` prefers CUDA, then MPS, then CPU.

```sh
git clone https://github.com/codewithlaurin/I.BA_NLP.H2601.git
cd I.BA_NLP.H2601
uv sync --locked
```

Download and decompress the pretrained English FastText binary into `data/embeddings/`:

```sh
mkdir -p data/embeddings
cd data/embeddings
uv run --project ../.. python -c "import fasttext.util; fasttext.util.download_model('en', if_exists='ignore')"
cd ../..
```

The download requires several GB of disk space and sufficient memory to load the binary. The dataset is downloaded automatically on first use. Pretrained binaries, generated checkpoints, and local experiment outputs are excluded from Git.

## Training

Run commands from the repository root.

To inspect data preparation without loading FastText or training a model:

```sh
uv run python train_classifier.py --batch-size 64 --device cpu
```

This loads train/validation, builds the vocabulary, and prints split and batch shapes. Training requires either `--fasttext-model` or `--inputs-path`; `train_rnn.py` requires one of these sources on every invocation.

The following commands use the loss-selected hyperparameters reported below and start fresh training with tracking disabled.

### Mean pooling

```sh
uv run python train_classifier.py \
  --fasttext-model data/embeddings/cc.en.300.bin \
  --hidden-dim 512 --dropout 0.4 \
  --learning-rate 0.003 --weight-decay 0.0001 \
  --batch-size 256 --epochs 60 --patience 8 --min-delta 0 \
  --seed 42 --save-accuracy-checkpoint \
  --wandb-mode disabled
```

### GRU

```sh
uv run python train_rnn.py \
  --fasttext-model data/embeddings/cc.en.300.bin \
  --rnn-type gru --rnn-hidden-size 256 --rnn-dropout 0.2 \
  --hidden-dim 128 --dropout 0.2 \
  --learning-rate 0.006 --weight-decay 0.0001 \
  --batch-size 256 --gradient-clip-norm 1.0 \
  --epochs 60 --patience 8 --min-delta 0 \
  --seed 42 --save-accuracy-checkpoint \
  --wandb-mode disabled
```

Use `--rnn-type lstm` to train an LSTM. Add `--bidirectional` for a bidirectional encoder; a width of 128 per direction produces 256 features per field. These flags change the architecture and do not imply that the GRU's selected hyperparameters are optimal for the other variants.

The commands above construct a training-only vocabulary with UNK fallback. The reported mean-pooling experiments instead used a frozen lookup cache extended with pretrained vectors for validation-only word types, without using labels. Exact reproduction of those inputs requires the original saved `inputs.pt`, available with the corresponding model artifact when uploaded. Replace `--fasttext-model ...` with `--inputs-path /path/to/inputs.pt` to use that cache. The reported GRU uses the original training-vocabulary/UNK policy. Reusing inputs preserves vectors and token IDs but still initializes fresh trainable weights.

Both models use AdamW, with decay on weights and zero decay on biases. Validation runs once per epoch with dropout disabled and no gradient updates. The lowest validation loss selects `best.pt`; patience also uses validation loss. `--min-delta` controls the improvement needed to reset patience, while every new minimum loss still saves `best.pt`. No learning-rate scheduler or label smoothing is used. Seeds control initialization and shuffling but do not guarantee identical results across devices or library versions.

Use `--learning-rates 0.0003 0.001 0.003` instead of a single `--learning-rate` for a sequential comparison. Each trial starts with fresh trainable weights and is compared by validation loss.

The commands above override these CLI defaults:

| Option | Mean pooling | RNN |
| --- | --- | --- |
| Batch size | `auto` | 64 |
| Classifier width / dropout | 128 / 0.2 | 256 / 0.2 |
| Learning rate / weight decay | 0.001 / 0.0001 | 0.001 / 0.0001 |
| Epoch cap / patience / min delta | 30 / 5 / 0 | 30 / 5 / 0 |
| Recurrent settings | N/A | Forward GRU, width 128, dropout 0 |
| Gradient clipping | None | Norm 1.0; disable with `--no-gradient-clipping` |

Both CLIs default to seed 42, device `auto`, online W&B tracking, and no accuracy checkpoint.

### Automatic batch sizing

During training, `--batch-size auto` probes worst-length batches on CUDA/MPS using a disposable model and a complete forward/backward/optimizer step. `--memory-margin` defaults to 0.2, and `--max-batch-size` optionally caps the search. CPU uses a fallback of 64 examples, subject to the cap and training-set size. The largest fitting batch is not necessarily the best optimization setting.

To retrain a saved configuration with automatic sizing:

```sh
uv run python train_auto_batch.py \
  --reference-checkpoint outputs/experiments/EXPERIMENT/TRIAL/best.pt \
  --memory-margin 0.2 --max-batch-size 256 \
  --seed 42 --device auto --wandb-mode disabled
```

Replace the checkpoint path with an existing mean-pooling or recurrent checkpoint. The wrapper reuses its hyperparameters and matching frozen inputs, initializes fresh trainable weights and optimizer state, and writes under `outputs/auto-batch/` by default. It does not resume training, evaluate test data, or enable accuracy-checkpoint saving.

All options are available through:

```sh
uv run python train_classifier.py --help
uv run python train_rnn.py --help
uv run python train_auto_batch.py --help
```

## Evaluation and checkpoints

The mean-pooling and RNN entry points create a unique directory under `outputs/experiments/` by default; `--output-dir` changes the parent directory:

```text
<experiment>/
├── inputs.pt              # Frozen embeddings, vocabulary, and input metadata
├── summary.json           # Trial comparison and selected trial
└── <trial>/
    ├── config.json
    ├── metrics.jsonl
    ├── summary.json
    ├── wandb.json         # Run identity; IDs/URLs depend on tracking mode
    ├── best.pt            # Minimum validation-loss checkpoint
    ├── best_accuracy.pt   # Optional maximum validation-accuracy checkpoint
    └── last.pt            # Latest completed epoch
```

`best_accuracy.pt` is saved only with `--save-accuracy-checkpoint`; accuracy ties use lower loss, then the earlier epoch. Early stopping remains loss-based. `best_validation_accuracy` in summaries means accuracy at `best.pt`; with the accuracy-checkpoint flag, summaries also include `peak_validation_accuracy`, the highest logged accuracy.

Keep a checkpoint together with its experiment's `inputs.pt`, preserving their relative directory structure. Restoration does not need the original FastText binary. Checkpoints contain optimizer state, but automatic CLI training resume and exact RNG-state replay are not implemented.

Evaluate existing checkpoints through the shared Python API. Run these snippets from the repository root in the project environment, for example with `uv run python`. Replace the checkpoint path with the file to evaluate.

For a recurrent checkpoint:

```python
from src.art_reasoning.batching import make_dataloader
from src.art_reasoning.data import load_experiment_data
from src.art_reasoning.training import load_trained_rnn, run_epoch

model, vocabulary = load_trained_rnn(
    "outputs/experiments/EXPERIMENT/TRIAL/best.pt", device="cpu"
)
data = load_experiment_data(vocabulary=vocabulary, include_test=True)
metrics = run_epoch(None, model, make_dataloader(data.test, batch_size=64), "cpu")
print(metrics)
```

For a mean-pooling checkpoint:

```python
from src.art_reasoning.batching import make_dataloader
from src.art_reasoning.data import load_experiment_data
from src.art_reasoning.training import load_trained_classifier, run_epoch

inputs, classifier, vocabulary = load_trained_classifier(
    "outputs/experiments/EXPERIMENT/TRIAL/best.pt", device="cpu"
)
data = load_experiment_data(vocabulary=vocabulary, include_test=True)
metrics = run_epoch(
    inputs, classifier, make_dataloader(data.test, batch_size=64), "cpu"
)
print(metrics)
```

These snippets print loss, accuracy, and example count without creating a W&B run or saving an evaluation report. There is currently no standalone evaluation CLI in the repository.

Adding `--evaluate-test` to either training command trains first, then evaluates the loss-selected checkpoint of the selected trial. Keep this flag off during tuning. Use validation metrics for selection and reserve test metrics for reporting.

## Results

Recorded on 2026-11-01. The primary configuration-selection criterion is validation loss.

### Loss-selected configurations

| Setting | Mean pooling | Forward GRU |
| --- | --- | --- |
| Frozen input policy | Extended FastText lookup cache | Training vocabulary / UNK |
| Recurrent width / dropout | N/A | 256 / 0.2 |
| Classifier width / dropout | 512 / 0.4 | 128 / 0.2 |
| Batch size | 256 | 256 |
| Learning rate | 0.003 | 0.006 |
| Weight decay | 0.0001 | 0.0001 |
| Epoch cap / patience | 60 / 8 | 60 / 8 |
| Gradient clipping | None | Norm 1.0 |

Validation results below average the three new seeds 101, 202, and 303. The screening seed 42 is excluded. Standard deviations are sample SD across training seeds on the same validation set.

| Model | Mean best validation loss ± SD | Mean accuracy at the loss-selected checkpoint |
| --- | --- | --- |
| Mean pooling | 0.669486 ± 0.001137 | 59.00% |
| GRU | 0.589452 ± 0.003696 | 67.97% |

The best individual loss-selected checkpoints remained the seed-42 screening winners. Their test evaluations used the same 1,532 examples:

| Model | Validation loss | Validation accuracy | Test loss | Test accuracy |
| --- | --- | --- | --- | --- |
| Mean pooling | 0.663558 | 58.7% | 0.796570 | 50.98% |
| GRU | 0.570383 | 67.0% | 0.887301 | 53.66% |

These are single-checkpoint test results, not three-seed test averages.

### Additional accuracy-selected evaluation

A separate comparison evaluated accuracy-selected checkpoints at seeds 101, 202, and 303 for the configurations with the highest mean peak validation accuracy. Mean pooling used head 512, dropout 0, batch 64, LR 0.003, and decay 0; the GRU used the loss-selected configuration above. Although its configuration originated from loss selection, the GRU epochs evaluated here were selected by accuracy.

| Model | Mean validation accuracy | Mean test accuracy ± SD | Mean test loss |
| --- | --- | --- | --- |
| Mean pooling | 59.80% | 50.81% ± 0.19 pp | 0.834773 |
| GRU | 69.27% | 53.26% ± 0.30 pp | 1.676421 |

The GRU has higher validation and test accuracy in these comparisons, but both models show a substantial validation-test gap. Aggregate scores do not establish its cause. A different accuracy-origin GRU configuration reached 70.3% on the screening seed but only 50.8% at seed 303; the historical 70.3% epoch's weights were not retained. Reported peaks, saved-checkpoint metrics, and across-seed averages should therefore be distinguished.

Some follow-up experiments were designed after earlier test evaluations had been inspected. Their configuration selection continued to use validation metrics; the chronology and earlier test results remain part of the experiment record.

### Initial expectations

Before experimentation, expected best test accuracies were 55-60% for mean pooling and 70-75% for the RNN, against a 50% random reference. The expectation was that pretrained semantic representations would offer a modest baseline and that sequence encoding would help capture word order. The reported results fell below these predictions.

## Experiment tracking

The [saved W&B view](https://wandb.ai/laurin-uni-workspace/nlp-classifier?nw=5bi9k4rwebs) contains 23 selected training runs, with seven core runs initially visible. It covers LSTM, GRU, BiLSTM, BiGRU, the loss/accuracy winners, and matched learning-rate, dropout, classifier-width, and batch-size comparisons. Runs are sorted by best validation loss ascending, breaking ties by peak validation accuracy descending. Training curves use mild smoothing; validation curves are unsmoothed. The test table is a static reporting snapshot.

The broader search included learning-rate/dropout screening, OOV-policy comparisons, bidirectional variants, automatic batch sizing, two 162-configuration grids, and 12 finalist runs on new seeds. Architecture representatives have different tuned settings; batch comparisons have different update counts under the same epoch cap. These comparisons do not isolate architecture or batch noise alone.

To log new runs to your own account:

```sh
uv run wandb login --verify
```

Replace `--wandb-mode disabled` in a training command with:

```text
--wandb-mode online --wandb-entity YOUR_ACCOUNT_OR_TEAM --wandb-project YOUR_PROJECT --no-wandb-log-model
```

Set entity/project explicitly because the code defaults to the author's course workspace. Use `--wandb-mode offline` for later synchronization. Local JSON metrics are saved in every mode. The canonical `model_family` values are `mean_pool` and `rnn`; cell type and direction are separate configuration fields.

Include `--no-wandb-log-model` for online/offline training with the current code: the training loop passes an `accuracy_checkpoint` keyword that the artifact-upload helper does not accept, so default artifact upload fails at the end of a trial. Local checkpoints are still written. Disabled tracking skips this upload path.

### Sweep configurations

`docs/` contains W&B sweep YAML files. `docs/sweep_classifier.yaml` uses the downloadable FastText binary and the current mean-pooling entry point. The RNN, overnight, and automatic-batch sweeps reference ignored local input caches or checkpoints; update those paths, device settings, and W&B account/project before use. Add `--no-wandb-log-model` to their training commands while the upload limitation above remains.

`docs/sweep_finalist_seeds.yaml` and `docs/sweep_lr_001.yaml` are historical configurations: their programs, `train_finalist_seeds.py` and `train_learning_rate_sweep.py`, are no longer included. Recreate their runs through the supported training entry points instead.

## Local checks

Run the synthetic checks for saved inputs, checkpoint restoration, and automatic-batch dispatch without downloading the dataset or FastText model:

```sh
uv run python -m unittest discover -s tests -v
```

## Repository layout

```text
src/art_reasoning/      Data preparation, batching, embeddings, models, training, tracking
train_classifier.py    Mean-pooling training and learning-rate comparisons
train_rnn.py           GRU/LSTM training with optional bidirectionality
train_auto_batch.py    Fresh training from a reference config with auto batch sizing
tests/                 Synthetic saved-input and automatic-batch checks
docs/                  Current and historical W&B sweep configurations
EXPECTATIONS.md        Predictions recorded before the experiments
pyproject.toml         Python requirements and dependencies
uv.lock                Locked dependency versions
data/embeddings/       Downloaded pretrained vectors (not versioned)
outputs/               Generated experiments and evaluations (not versioned)
```

A fresh clone does not contain the original frozen input caches or trained checkpoints. Obtain the matching W&B artifact when available, or run training to generate new artifacts. The recorded scores refer to the original configurations, inputs, and environments.

## References and tools

- Bhagavatula et al. (2020), [Abductive Commonsense Reasoning](https://arxiv.org/abs/1908.05739): the original ART task.
- [Course-specific ART dataset](https://huggingface.co/datasets/amarfurt/art-nlp-hs26): the splits used in this project.
- [FastText pretrained vectors](https://fasttext.cc/docs/en/crawl-vectors.html): the frozen English embeddings.
- [PyTorch](https://pytorch.org/), [Hugging Face Datasets](https://huggingface.co/docs/datasets/), and [Weights & Biases](https://wandb.ai/): implementation, data loading, and experiment tracking.
- OpenAI Codex assisted with code, experiment tooling, documentation, and review.
