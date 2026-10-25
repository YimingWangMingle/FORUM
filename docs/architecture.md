# Architecture and data flow

## Training stages

```text
Training sessions only
  -> causal histories + admissible same-action future targets     data.py
  -> future-aware encoder / EMA target                          representation.py
  -> freeze encoder
  -> session bootstrap world-model ensemble                     world.py
       feedback -> termination -> nonterminal displacement       diffusion.py
  -> freeze world models
  -> initialize raw critic, then fit behavior support model     policy.py
  -> start short rollouts at logged histories
       policy over supported legal items                        support.py
       within-member averaging, then across-member disagreement uncertainty.py
       one fixed model member per trajectory
  -> raw-reward critic / penalized critic / actor updates        policy.py
  -> export completed frozen checkpoints                        train.py
```

Future feedback supervises the representation loss, but is never part of the
current history input. World-model training conditions on the current logged
feedback; simulated dynamics condition on generated feedback.

In training logs, only observed terminal events set `done=True`. A separate
`terminal_observed` mask excludes censored final rows from departure and dynamics
supervision. Missing future observations exclude incomplete representation
windows; an unknown boundary survival masks the successor-distance term.
During simulation, `model_done` and `catalog_exhausted` are recorded separately;
their union ends the trajectory and zeros both TD tails. Nonterminal horizon
truncations retain value bootstrapping. A terminal successor is absorbing, with
zero continuation contribution.

The uncertainty estimator uses the target **raw** critic. The penalized critic
is separate, so the risk signal does not recursively depend on its own penalty.
Both target critics are fixed during a policy update and refreshed afterward.

## Deployment and loading

`serving.py` loads encoder, behavior model and actor from `serving.pt`. It encodes
observed history, constructs supported legal candidates, and returns actor-ranked
items. It performs no diffusion simulation or critic evaluation.

`checkpoint.py` loads the full research bundle from `training.pt`, including all
world models, raw/penalized critics and target critics. Optimizer/RNG state is not
saved, so this is model loading rather than exact interrupted-training resume.

## Run artifacts

The runtime creates input hashes, a source snapshot, configuration, bootstrap
session membership, training diagnostics, final rollouts and checkpoints in the
requested output directory. These generated artifacts are ignored by Git and are
not part of the distributed repository. The evaluator is a separate command;
it checks completion and checkpoint identity before loading evaluation rewards.

## Review entry points

Read `data.py` and `representation.py` for future supervision; `world.py` and
`diffusion.py` for generative modeling; `uncertainty.py` for risk; `policy.py` for
the actual actor/critic updates. `train.py` connects the full training sequence.
Tests exercise both individual equations and the assembled pipeline.
