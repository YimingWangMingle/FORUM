# Paper alignment and explicit implementation choices

Source: *Beyond Reward Accuracy: Latent World Modeling for Offline Reinforcement
Learning in Recommendation*, supplied 11-page manuscript, Sections 4.1–4.4,
Algorithm 1 and Section 5.2. Performance tables in that draft contain placeholders.
This repository implements the training algorithm; it does not substantiate those
placeholder performance claims or a four-dataset benchmark reproduction.

## Equation mapping

| Paper | Implementation |
|---|---|
| Eq. 5: causal history encoder | `networks.py:HistoryEncoder`, `data.py:HistoryTable` |
| Eq. 6: observed return and survival length | `data.py:window_target` |
| Eqs. 7–8: same-action behavioral distance and loss | `representation.py:behavioral_target`, `train_representation` |
| Eq. 9: feedback followed by conditional termination | `world.py:feedback_distribution`, `losses`, `simulate` |
| Eqs. 10–11: conditional nonterminal displacement diffusion | `diffusion.py:DisplacementDiffusion` |
| Eq. 12: ensemble world-model loss | `world.py:train_ensemble` |
| Eqs. 13–14: average within each member, then compare members | `uncertainty.py:estimate_uncertainty` |
| Eq. 15: local value-direction interpretation | Explanatory relation; not an additional training objective |
| Eq. 16: reward and continuation-value risk penalty | `uncertainty.py:disagreement` |
| Eq. 17: supported recommendation distribution | `support.py:support_mask`, `policy.py:action_distribution` |
| Eq. 18: uncertainty-adjusted short trajectories | `policy.py:collect_rollouts` |
| Eqs. 19–20: separate raw and penalized TD targets / A2C | `policy.py:td_targets`, `actor_critic_losses`, `fit_policy` |
| Algorithm 1: staged training and freezing | `train.py:run_training` |
| Deployment without world-model simulation | `serving.py:Recommender` |

## Stated settings preserved in `configs/paper.json`

Gamma 0.99; history window 20; latent dimension 64; horizons {1,3,5}; length weight
0.5; successor weight 0.1; encoder EMA 0.995; ensemble size 5; diffusion steps 20;
four uncertainty samples per member; reward/value risk weights 0.1; support K=50
and threshold 1e-5; simulated horizon H=5; two-layer 256-unit actor/critics;
three-layer 256-unit denoiser; Adam learning rate 3e-4; target update rate 0.005;
batch size 256; 50 representation sampling epochs and 50 world-model epochs.
A world-model epoch visits every row of that member's session bootstrap once.
A representation epoch draws `number_of_training_rows` pairs from the joint
pair/horizon sampler; the manuscript does not define an exhaustive pair epoch.

The paper's diffusion N and ensemble M are `diffusion_steps` and `ensemble_size`.
They are distinct from the optional Coat environment's N=4, M=0 and T=30.

## Details chosen for this implementation

- A user-conditioned GRU encodes history; scalar feedback uses linear+tanh
  embeddings. The item/user embedding width is an explicit engineering default.
- Networks use SiLU and gradient norm clipping at 10. Actor initialization copies
  the fitted behavior policy; the penalized critic starts from the raw critic.
- Conditional DDPM uses a cosine schedule in the frozen encoder's original latent
  coordinates, with no displacement normalization or clipping. The Eq. (11) loss
  sums squared errors over latent dimensions and averages over observations.
- Same-action ordered pairs are sampled uniformly, then a horizon is drawn
  uniformly from the selected pair's admissible horizons, following Eq. (8).
  Nested horizon eligibility avoids materializing the Cartesian product.
- World-model epochs shuffle the session-bootstrap rows, retaining bootstrap
  multiplicities. Diffusion batches independently sample observed nonterminal
  rows from that same bootstrap, matching the conditional expectation over D_nt.
  Smoke/quick configurations use explicit update budgets with replacement.
- The session schema permits right censoring. At the last censored observation,
  termination is unknown: feedback remains supervised, but termination likelihood,
  diffusion supervision and the representation successor term are masked. This
  is an explicit missing-label convention for the session adapter, not a claim
  that an unknown departure equals nontermination.
- Feedback and termination likelihood weights default to 1. The paper does not
  specify behavior/value initialization or policy update budgets; their default
  1,000 updates in the paper-scale preset are implementation choices.
- Raw-critic initialization uses observed suffix returns, with a target-value
  bootstrap for censored boundaries. This specific censor treatment is an explicit
  extension to the manuscript's unspecific logged-return initialization.
- Item and feedback embeddings are shared from the frozen encoder. Ensemble
  feedback, termination and denoising predictors are independently fitted on
  session-level bootstrap samples.
- When NX_0 is enabled, support selection first excludes seen items, then applies
  top-K and the threshold using full-catalog behavior probabilities. Fallback is
  the highest-probability legal item. Critics retain the paper's `V(z)` and `W(z)`
  inputs rather than adding the external seen-item mask. Exhausting the catalog
  is an external terminal boundary: both TD tails and continuation uncertainty
  become zero. Saved rollouts distinguish model termination from catalog exhaustion.
- Member standard deviation uses the population convention for a uniformly
  sampled member. Finite within-member Monte Carlo samples still add sampling
  noise; the resulting uncertainty is a risk proxy, not a calibrated error bound.
- Feedback is scalar and `rho(y)=y`. Categorical ratings and Gaussian feedback are
  supported. Multi-dimensional feedback, cold-start embeddings and specialized
  raw-file adapters for KuaiRec/KuaiRand/Yahoo are not implemented here.

## Reviewable invariants

The encoder never consumes current or future feedback as input. Censored future
windows are excluded rather than padded with invented rewards. The encoder and
world models are frozen before policy learning. A simulated trajectory keeps its
member fixed; uncertainty still considers all members. The raw critic receives
unpenalized rewards, and only the separate critic/actor optimize penalized reward.
Targets remain fixed during an update. Generated termination prevents diffusion
and continuation; rollout length truncation retains bootstrapping.

Tests cover these invariants, formula values, gradients, save/load and the complete
pipeline. Functional tests establish implementation behavior, not benchmark gains.

## Verification and remaining scope

See [Implementation Audit](implementation_audit.md) for corrected mismatches,
regression evidence and numerical limitations observed in short runs. Old
normalized-diffusion training checkpoints must be retrained and are rejected by
the full-bundle loader. These changes do not validate the manuscript's experimental
sections, theoretical assumptions, unimplemented ablations or benchmark claims.
