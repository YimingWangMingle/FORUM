import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch


def seed_all(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def freeze(model):
    model.eval()
    model.requires_grad_(False)
    return model


def training_batches(rows, rng, batch_size, steps, epochs):
    """Explicit step budgets sample with replacement; epochs traverse the bootstrap."""
    if steps:
        for _ in range(steps):
            yield rng.choice(rows, batch_size)
    else:
        for _ in range(epochs):
            shuffled = rng.permutation(rows)
            for start in range(0, len(rows), batch_size):
                yield shuffled[start:start+batch_size]


@torch.no_grad()
def soft_update(target, source, tau):
    for dest, src in zip(target.parameters(), source.parameters()):
        dest.lerp_(src, tau)


def optimizer_step(loss, optimizer, parameters):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def log(stage, **values):
    print(json.dumps(dict(stage=stage, **values), allow_nan=False), flush=True)
