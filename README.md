<div align="center">

# FORUM

### Future-Oriented Representation and Uncertainty Modeling

**Latent world models for offline reinforcement learning in recommendation**

[Overview](#overview) · [Installation](#installation) · [Quick Start](#quick-start) · [Training](#training) · [Evaluation](#evaluation) · [Documentation](#documentation)

</div>

This repository provides a PyTorch implementation of **FORUM**, the method described in
*Beyond Reward Accuracy: Latent World Modeling for Offline Reinforcement Learning in Recommendation*.

## Overview

FORUM learns recommendation policies from logged interactions through a latent user
world model. Its representation objective distinguishes histories with different
multi-step feedback and engagement outcomes. An ensemble of conditional diffusion
models predicts user responses and latent transitions, while disagreement in rewards
and continuation values guides conservative policy optimization.

The framework consists of four components:

- **Future-aware behavioral representation.** An action-matched, multi-horizon objective
  trains a causal history encoder using observed returns, engagement lengths, and
  successor-state distances.
- **Action-conditioned world modeling.** Session-bootstrap ensemble members model
  feedback, termination, and nonterminal latent displacements through conditional diffusion.
- **Value-sensitive uncertainty.** Predictions are averaged within each member before
  measuring disagreement across members in immediate rewards and continuation values.
- **Conservative policy learning.** A2C optimizes short, behavior-supported rollouts
  with uncertainty-adjusted rewards and separate raw-return and penalized-return critics.

At deployment, the encoder and actor rank behavior-supported candidates directly.
World-model simulation and critic evaluation are used only during training.

## Installation

Run the following commands from the repository root. The locally tested environment
uses **Python 3.10**, **PyTorch 2.5.1**, and **NumPy 1.23.5** on CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The installed `forum` command and `python -m forum` expose the same interface.
Training accepts `--device cpu`, `--device cuda`, or `--device mps`; the examples
below use the validated CPU path.

## Quick Start

Run a small end-to-end experiment without downloading a dataset:

```bash
python -m forum smoke \
  --config configs/smoke.json \
  --out runs/smoke \
  --device cpu
```

Generate recommendations from the resulting policy:

```bash
python -m forum recommend \
  --checkpoint runs/smoke/serving.pt \
  --user 0 \
  --history '[[12,4.0],[8,3.0]]' \
  --count 5
```

Each history entry is an `[item_id, observed_feedback]` pair. With the default NX_0
constraint, previously recommended items are excluded from the candidate set.
The smoke experiment uses synthetic sessions to verify the training and inference pipeline.

## Data Preparation

The general training interface accepts **JSONL sessions**, with one session per line:

```json
{"session_id":"u0-s0","user":0,"actions":[1,4,2],"feedback":[5.0,2.0,4.0],"terminated":false}
```

User and item IDs are zero-based integers. Actions and feedback follow their observed
order, and `terminated` distinguishes an observed terminal event from a logging cutoff.
Define the training, validation, and test partitions before constructing examples.
See [Data Format](docs/data_format.md) for field definitions, vocabulary requirements,
feedback distributions, and censoring semantics.

A synthetic dataset is provided through a generation script:

```bash
python scripts/make_toy_data.py --out data/toy.jsonl
```

This produces 32 sessions for 16 users over a catalog of 40 items. Both observed
terminations and censored sessions are included.

## Training

### Training from interaction sessions

Train through the standard data interface using the generated dataset:

```bash
python -m forum train \
  --sessions data/toy.jsonl \
  --n-users 16 \
  --n-items 40 \
  --feedback-values 1,2,3,4,5 \
  --config configs/quick.json \
  --out runs/quick \
  --device cpu
```

For a real dataset, replace the session path and vocabulary sizes. The
`--feedback-values` option specifies a categorical feedback distribution; omitting
it selects a Gaussian scalar distribution. The current reward transform is `rho(y) = y`.

### Configuration presets

| Configuration | Latent / hidden dimension | Ensemble members | Diffusion steps | Purpose |
|---|---:|---:|---:|---|
| [`smoke.json`](configs/smoke.json) | 16 / 32 | 2 | 4 | Minimal functional validation |
| [`quick.json`](configs/quick.json) | 32 / 64 | 3 | 8 | Short exploratory training |
| [`paper.json`](configs/paper.json) | 64 / 256 | 5 | 20 | Manuscript-scale model settings |

All presets use future-supervision horizons `{1, 3, 5}`. Training budgets and model
parameters are defined in the configuration files. Short presets check execution;
they do not establish world-model or policy convergence. Settings supplied by the manuscript
and additional implementation choices are distinguished in
[Paper Alignment](docs/paper_alignment.md).

### Coat adapter

An adapter is available for the Coat training matrix:

```bash
python -m forum train-coat \
  --train-matrix /path/to/Coat/train.ascii \
  --config configs/quick.json \
  --out runs/coat \
  --device cpu
```

Coat provides static ratings rather than timestamped sessions. The adapter applies a
fixed per-user ordering and treats the end of observed ratings as censoring. This
adaptation is documented separately from the chronological-session interface in
[Data Format](docs/data_format.md).

### Checkpoints

Each experiment writes to a new output directory and produces:

- `serving.pt`: the history encoder, behavior support model, and recommendation actor.
- `training.pt`: all learned components, including the world-model ensemble and both
  critics with their target networks.
- Configuration, input fingerprints, source snapshots, training diagnostics, and
  bootstrap-session records for inspecting the run.

Existing run directories are not overwritten. The full checkpoint supports model
loading through `forum.checkpoint.load_training_bundle`; optimizer and RNG states
are not included for exact interrupted-training resumption.

## Evaluation

The optional Coat evaluator loads a completed, frozen policy and checks the
fingerprint of the separately supplied RL4Rec completed rating matrix:

```bash
python -m forum evaluate-coat \
  --run runs/coat \
  --matrix /path/to/coat_pseudoGT_ratingM.ascii \
  --out runs/coat/evaluation.json
```

Evaluation uses **N=4, M=0, T=30, and NX_0**, with all 290 users evaluated once.
The output includes episode returns, average reward, episode lengths, and per-user
action sequences. The completed matrix defines a simulation environment; its protocol
and interpretation are described in [Data Format](docs/data_format.md#optional-coat-adapter).

The current release includes functional validation and a Coat adapter. Benchmark
performance results and raw-data adapters for KuaiRec, KuaiRand, and Yahoo are not
included.

## Repository Structure

```text
FORUM-github/
├── configs/                 # Training presets
├── forum/
│   ├── data.py              # Sessions, causal histories, and future targets
│   ├── networks.py          # History encoder, actor, and critics
│   ├── representation.py    # Future-aware representation learning
│   ├── diffusion.py         # Conditional latent-displacement diffusion
│   ├── world.py             # Feedback, termination, and ensemble fitting
│   ├── uncertainty.py       # Reward and continuation-value disagreement
│   ├── support.py           # Behavior support and action constraints
│   ├── policy.py            # Simulated rollouts and dual-critic A2C
│   ├── train.py             # End-to-end training
│   ├── serving.py           # Recommendation inference
│   ├── checkpoint.py        # Full-model loading
│   └── evaluate.py          # Frozen-policy evaluation
├── scripts/                 # Synthetic data generation
├── tests/                   # Unit and integration tests
├── docs/                    # Architecture, data interface, and paper mapping
└── .github/workflows/       # Continuous integration
```

## Testing

```bash
python -m unittest discover -s tests -v
```

Tests cover future-target construction, censoring, gradient propagation, diffusion,
ensemble uncertainty, support constraints, dual-critic targets, checkpoint loading,
and the public training and recommendation commands. The GitHub Actions workflow
is configured for Python 3.10 and 3.11 on Linux CPU.

## Documentation

- [Architecture](docs/architecture.md): component interfaces and the complete data flow.
- [Data Format](docs/data_format.md): session schema, preprocessing requirements, and evaluation inputs.
- [Paper Alignment](docs/paper_alignment.md): equation-to-code mapping and implementation details.
- [Implementation Audit](docs/implementation_audit.md): verified equations, corrected mismatches, regression tests, and validation limits.

## Acknowledgements

FORUM is implemented with [PyTorch](https://pytorch.org/) and [NumPy](https://numpy.org/).
The documentation structure draws on [EasyRL4Rec](https://github.com/chongminggao/EasyRL4Rec)
and [EME](https://github.com/YimingWangMingle/EME).
Code and data provenance are described in [NOTICE](NOTICE).

## License

This project is released under the [MIT License](LICENSE).
