# Training data format

## JSONL sessions

One line is one complete observed session:

```json
{"session_id":"u0-s0","user":0,"actions":[1,4,2],"feedback":[5.0,2.0,4.0],"terminated":false}
```

| Field | Meaning |
|---|---|
| `session_id` | Unique session identifier |
| `user` | Nonnegative integer, below `--n-users` |
| `actions` | Nonempty ordered item IDs, below `--n-items` |
| `feedback` | Finite scalar observations, aligned with actions |
| `terminated` | Boolean: true only for an observed terminal event at the last interaction |

Actions and feedback must have equal length. NX_0 defaults to true and rejects
repeated actions inside a training session. Repeated-user sessions are allowed
with different session IDs. Current feedback and action are excluded from the
current encoded history. Only the preceding `history_window` interactions enter
the GRU; the external no-repeat mask retains the complete preceding item set.

Split raw data before creating training examples, respecting the intended user,
session or temporal protocol. This repository accepts a training file; it does
not invent validation/test splits, tune on test data or select a best test epoch.
Use a fixed ID vocabulary across splits. User/item IDs with no training evidence
are not a validated cold-start capability of this implementation.

## Feedback distributions and reward

`--feedback-values 1,2,3,4,5` fits a categorical rating distribution. All training
feedback must belong to the declared set. Omitting this option uses a Gaussian
scalar feedback distribution. The reward transform is currently identity,
`rho(y)=y`. Dataset-specific transformations must be defined consistently before
constructing sessions. Multi-dimensional feedback is not supported by this adapter.

## Terminal events and censoring

`terminated=false` means observation stopped without an observed user departure.
Future windows shorter than the requested horizon are then unavailable. A complete
window ending at that boundary is usable; its observed successor can still be
encoded, but survival after that last feedback is unknown. The boundary row
still supervises feedback; its termination label and nonterminal diffusion target
are excluded, and its representation successor-distance term is masked. Interior
rows have observed continuation; a true terminal row has an observed departure.
For raw-value initialization, a censored observed suffix bootstraps at its final
observed state, except when NX_0 has exhausted the catalog. A true terminal suffix
has zero tail value. This bootstrap is an explicit initialization convention,
not an observed future return.

Do not label an arbitrary logging cutoff as a user termination. Do not use held-out
feedback to fill incomplete training trajectories. Terminal transitions train the
feedback/termination heads but do not train or sample a nonterminal diffusion target.

## Synthetic data

`python scripts/make_toy_data.py --out data/toy.jsonl` creates 32 synthetic sessions,
16 users and a catalog of 40 items. It includes both terminal and censored sessions.
Files are created exclusively; the script refuses to overwrite an existing file.
This fixture exercises the ordinary JSONL CLI and is not a recommendation benchmark.

## Optional Coat adapter

`train-coat` accepts only the explicit 290x300 `train.ascii` matrix, with ratings
in 1..5 and zero for missing observations. It does not open sibling test files.
The default fixed ordering seed is 20261024. The archived input has 24 ratings
per user: 6,960 interactions in total, without timestamps or observed departures.
The adapter therefore constructs artificial histories and marks all session ends
as censored. It does not establish the paper's real future-behavior claims.

`evaluate-coat` is separate and requires a completed frozen run. It checks the
published completed RL4Rec matrix's SHA256:

```text
fee71820f155a3f43ab50858734a23102c8328cf9bd0e586ed5a7db2fb084223
```

Evaluation uses N=4, M=0, T=30 and NX_0; each of the 290 users runs once. At M=0,
nonnegative item distances cannot trigger a strict distance-below-M departure.
This is a completed simulator, not a matrix of wholly observed human ratings.
Matching these settings does not align every aspect of other papers' training
procedures. Prior repeated use of this public simulator also prevents an
untouched-test claim. The repository does not distribute its underlying data.
