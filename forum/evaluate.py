"""Read-only frozen-policy Coat simulator evaluation; never called by training."""
from pathlib import Path
import numpy as np

from .serving import Recommender
from .utils import sha256, write_json


COAT_GT_SHA256 = "fee71820f155a3f43ab50858734a23102c8328cf9bd0e586ed5a7db2fb084223"


def evaluate_coat(run, matrix_path, output, device="cpu"):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(output)
    import json
    summary = json.loads((run/"summary.json").read_text())
    if summary["status"] != "passed" or sha256(run/"serving.pt") != summary["checkpoints"]["serving.pt"]:
        raise ValueError("Checkpoint must match a completed frozen training run")
    # Only after completed training and checkpoint validation is the matrix parsed.
    if sha256(matrix_path) != COAT_GT_SHA256:
        raise ValueError("Unexpected Coat evaluation matrix hash")
    model = Recommender(run/"serving.pt", device)
    if model.n_items != 300 or model.n_users != 290 or not model.config.no_repeat:
        raise ValueError("Coat evaluator requires 290 users, 300 items and NX_0")
    matrix = np.loadtxt(matrix_path)
    if matrix.shape != (290, 300) or not np.isin(matrix, np.arange(1, 6)).all():
        raise ValueError("Invalid Coat simulator rewards")
    records = []
    for start in range(0, 290, 32):
        users = list(range(start, min(start+32, 290))); histories = [[] for _ in users]
        for turn in range(30):
            actions = model.rank(users, histories)
            for u, history, chosen in zip(users, histories, actions):
                a = chosen[0]
                if a in [x for x, _ in history]:
                    raise AssertionError("NX_0 violation")
                history.append((a, float(matrix[u, a])))
        for u, history in zip(users, histories):
            records.append(dict(user=u, actions=[a for a, _ in history], rewards=[r for _, r in history],
                                return_=sum(r for _, r in history), length=len(history)))
    result = dict(mean_return=float(np.mean([r["return_"] for r in records])), mean_length=30.,
        mean_reward=float(np.mean([r["return_"] for r in records])/30), users=records,
        protocol=dict(N=4, M=0, T=30, NX_0=True, evaluation_users="all 290 exactly once"),
        simulator="public RL4Rec completed pseudoGT, not fully observed human ratings",
        training_input=json.loads((run/"input.json").read_text()),
        checkpoint_sha256=summary["checkpoints"]["serving.pt"], evaluation_sha256=COAT_GT_SHA256,
        note="Exploratory result. Same public simulator has been used in earlier experiments; no untouched-test claim.")
    # At M=0, nonnegative item distances never trigger the strict distance < M exit.
    write_json(output, result)
    return {k: v for k, v in result.items() if k != "users"}
