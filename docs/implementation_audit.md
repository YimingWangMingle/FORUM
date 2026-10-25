# Implementation audit

Audit date: 2026-10-25. Scope: Sections 4.1–4.4, Algorithm 1, and the model settings
in Section 5.2 of *Beyond Reward Accuracy: Latent World Modeling for Offline
Reinforcement Learning in Recommendation* (the supplied 11-page manuscript).

Manuscript SHA256:

```text
09b22dbee469cbe0e0e1521b9bb8d0975e8d978b8b1aa488d4b647b4b793d070
```

The four method components are connected to the executable training path. They
are trained or evaluated during `forum train`, not merely declared as standalone
classes. This audit verifies implementation behavior; it does not establish the
manuscript's benchmark claims, theoretical assumptions, or policy convergence.

## Component and loss verification

| Component | Executed computation | Evidence |
|---|---|---|
| Future-aware representation | Strictly preceding history; same initial action; observed multi-step return and length; EMA successor distance; detached target; encoder frozen afterward | `data.py`, `representation.py`; numerical Eq. (7) check, censoring and pair-distribution tests, frozen-encoder pipeline checks |
| Latent world modeling | Independent session-bootstrap members; feedback likelihood; feedback-conditioned departure likelihood; nonterminal displacement DDPM; generated feedback conditions simulated transitions | `world.py`, `diffusion.py`; nonzero predictor gradients, exact forward-noise and reverse-step tests, generated-feedback conditioning test |
| Value-sensitive uncertainty | Within-member Monte Carlo mean, then population standard deviation across members; separate reward and raw-value continuation terms | `uncertainty.py`; exact disagreement arithmetic, within-member averaging and absorbing-continuation tests |
| Conservative policy | Shared behavior support in training and serving; one ensemble member per trajectory; risk-adjusted simulated reward; separate raw and penalized critics; detached A2C advantage | `support.py`, `policy.py`, `serving.py`; support and trajectory tests, numerical TD/loss/gradient checks, serving without world calls |

The implemented objectives are:

- Eq. (8): mean squared error between latent distance and the detached behavioral
  target. First sample an admissible ordered pair uniformly, then sample uniformly
  from that pair's admissible horizons.
- Eq. (11): expected **sum** of squared noise-prediction errors over latent
  coordinates. The clean diffusion variable is the detached raw displacement
  `z_next - z`, with no coordinate standardization.
- Eq. (12): diffusion loss plus weighted feedback and departure negative
  log-likelihoods. The diffusion expectation samples observed nonterminal rows
  from the same member's bootstrap, independently of the response minibatch.
- Eq. (16): `lambda_reward * U_reward + lambda_value * gamma * U_value`.
  The target raw critic supplies continuation values; the penalized critic never
  defines this penalty. Eq. (15) is an interpretation, not an additional loss.
- Eqs. (19)–(20): separate raw and penalized TD targets, both detached; two critic
  MSE losses; negative log probability multiplied by a detached penalized
  advantage. There is no additional CQL or policy-distillation objective.

## Corrections made during the audit

| Previous behavior | Corrected behavior | Reason |
|---|---|---|
| Diffusion standardized displacements | Diffusion uses raw frozen-encoder displacements | Matches the forward process in Eqs. (10)–(11) |
| Diffusion MSE averaged latent coordinates | Sum coordinates, average observations | Preserves Eq. (11)'s scale relative to the other Eq. (12) losses |
| Uniform admissible `(i,j,k)` triples | Uniform pair followed by uniform conditional horizon | Avoids overweighting pairs with more admissible horizons |
| Censored final row labeled as nontermination | Missing departure label is masked; boundary diffusion and successor supervision are excluded | An unobserved departure is not a confirmed continuation |
| Diffusion could be skipped when a response batch had only terminal rows | Independent nonterminal minibatch from the member's bootstrap | Keeps the dynamics objective an expectation over D_nt |
| Epoch setting only counted random replacement updates | World epochs shuffle and traverse bootstrap rows; representation epochs draw exactly one pair per training-row budget | Makes epoch and explicit-step budgets distinct and reproducible |
| Catalog exhaustion stopped rollout but retained a TD tail | Exhaustion zeros both TD tails and continuation uncertainty | No continuation action exists under NX_0 |
| Behavior fitting preceded raw-value initialization | Raw-value initialization precedes behavior fitting | Follows Algorithm 1's stage order |
| Terminal masking multiplied potentially unused nonfinite values by zero | Explicit selection discards unused TD tails | Preserves zero absorbing continuation |

Catalog exhaustion is an external NX_0 boundary, not a learned user-departure
label. Saved rollouts record `model_done` and `catalog_exhausted` separately.
A horizon-H cutoff with legal remaining actions remains nonterminal and retains
value bootstrapping. Legacy normalized-diffusion training bundles are rejected
by the full-model loader; retraining is required to use the audited algorithm.

## Data flow and gradient boundaries

1. The caller supplies the training partition. The loader builds causal history
   prefixes and separate future-supervision targets from that partition only.
   No validation/test file is discovered or opened by training.
2. The encoder learns from action-matched observed futures. The whole target is
   detached; the target encoder is an EMA. Censored incomplete windows are
   unavailable. At a complete window ending exactly at a censored boundary,
   return and length are available, but survival for the successor term is not.
3. The encoder is frozen. Each world member receives a whole-session bootstrap,
   including repeated sessions with their bootstrap multiplicities. Feedback
   embeddings and item vectors come from the frozen encoder. Departure and
   dynamics predictors use only observed labels.
4. The raw critic initializes from observed suffix returns. Censored suffixes
   use a target-value bootstrap, an explicit adapter convention; actual terminal
   suffixes and exhausted catalogs have zero tails. Behavior fitting uses logged
   state-item pairs. Actor initialization copies behavior; the penalized critic
   initializes from the raw critic.
5. Rollouts use generated feedback and generated transitions, with one fixed
   member per trajectory. Risk is recomputed using all members and the raw target
   critic. The ensemble, encoder, and target critics are fixed within an update.
6. The actor and both critics receive their own losses. Targets and actor
   advantages are detached. Target critics refresh only between updates. Saved
   weight comparisons verify that encoder and world weights did not change.
7. Serving reloads only encoder, behavior support model, and actor. It ranks legal
   supported items from observed histories without running the world or critics.

The optional Coat adapter has no timestamps or observed departures. Its fixed
artificial order is disclosed. Its 6,960 training ratings produce 6,670 interior
continuation labels and 290 unknown final departure labels. This is an adapter
check, not evidence for a genuine sequential Coat protocol. The optional evaluator
retains N=4, M=0, T=30 and NX_0; no evaluation matrix was used during this audit.

## Executed validation

Environment: local Mac, CPU, Python 3.10.21, PyTorch 2.5.1, NumPy 1.23.5.

| Validation | Result |
|---|---|
| Full unit, numerical-contract, integration and public-CLI suite | 52 tests passed |
| Synthetic smoke preset | 335 logged rows; 144 simulated transitions; 17/17 pipeline checks passed |
| Synthetic quick preset | 335 logged rows; 4,308 simulated transitions; 17/17 pipeline checks passed |
| Coat training-only smoke | 6,960 logged rows; 146 simulated transitions; 17/17 pipeline checks passed |
| Epoch-budget integration path | Representation, ensemble, raw/behavior initialization and policy stages completed |
| Training checkpoint reload | Full model tensors round-trip exactly |
| Deployment path | 30 no-repeat recommendations tested without world-model calls |
| Distribution package | Wheel built, installed in isolation, imported outside the source tree; installed-package smoke passed |

Final standalone runs reported approximately 0.90 s, 7.26 s and 1.42 s internally
for synthetic smoke, synthetic quick and Coat smoke, respectively. These are
functional-test timings, exclude interpreter startup, and are not estimates for
paper-scale training. Run directories retain configuration, training-input
fingerprints, source snapshots, bootstrap membership, losses and checkpoint hashes.
They are intentionally excluded from the source archive.

Reproduce the automated checks from the repository root:

```bash
python -m unittest discover -s tests -v
python -m forum smoke --config configs/smoke.json --out runs/audit-smoke
python -m forum smoke --config configs/quick.json --out runs/audit-quick
```

Each output directory must be new. The GitHub Actions workflow is provided but
was not executed remotely as part of this audit. CUDA and MPS were not validated.

## Remaining limits for scientific review

The short synthetic quick run is finite and completes, but it is **not converged**:
at policy update 100, raw-critic MSE is about 3.01e7 and mean risk penalty is about
107.2. The untrained/undertrained raw-displacement reverse process can generate
large latent states and continuation values. These observations preclude using
this run as evidence of stable value estimation or recommendation gains. No
clipping, normalization, held-out correction or changed objective was introduced
to conceal this behavior. Substantial world-model training and held-out predictive
diagnostics are still needed before making empirical claims.

The repository is suitable for examining and executing the four-component method.
It is not a complete reproduction of Sections 5.3–5.5: dataset-wide benchmark
results, all ablations and multi-step held-out diagnostics are not supplied. The
manuscript itself contains placeholder performance values. Hyperparameters not
specified by the manuscript and adapter conventions are enumerated in
[Paper Alignment](paper_alignment.md); data limitations are in
[Data Format](data_format.md).
