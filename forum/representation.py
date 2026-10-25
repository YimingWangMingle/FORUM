"""Eqs. (6)-(8): action-matched, multi-horizon behavioral metric training."""
from copy import deepcopy
import math
import numpy as np
import torch

from .networks import HistoryEncoder
from .utils import freeze, log, optimizer_step, soft_update


def behavioral_target(g_left, g_right, l_left, l_right, horizon, mask,
                      next_left, next_right, config):
    successor = torch.where(mask.bool(), torch.linalg.vector_norm(next_left-next_right, dim=-1), 0.)
    return ((g_left-g_right).abs() + config.lambda_length*(l_left-l_right).abs()/horizon
            + config.lambda_successor*config.gamma**horizon*mask*successor).detach()


def train_representation(table, config, device):
    encoder = HistoryEncoder(table.n_users, table.n_items, config).to(device)
    target = freeze(deepcopy(encoder))
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate)
    sampler = table.pairs(config.horizons, config.gamma)
    rng = np.random.default_rng(config.seed + 101)
    batches_per_epoch = math.ceil(len(table)/config.batch_size)
    steps = config.representation_steps or config.representation_epochs*batches_per_epoch
    history = []
    for step in range(1, steps+1):
        batch_size = config.batch_size
        if not config.representation_steps:
            batch_size = min(batch_size, len(table)-((step-1) % batches_per_epoch)*batch_size)
        pair = sampler.sample(rng, batch_size)
        z_i = encoder(*table.batch(table.state[pair["left"]], device))
        z_j = encoder(*table.batch(table.state[pair["right"]], device))
        with torch.no_grad():
            z_in = target(*table.batch(pair["next_left"], device))
            z_jn = target(*table.batch(pair["next_right"], device))
            tensors = [torch.as_tensor(pair[k], dtype=torch.float32, device=device) for k in
                       ("g_left", "g_right", "l_left", "l_right", "horizon", "mask")]
            delta = behavioral_target(*tensors, z_in, z_jn, config)
        loss = (torch.linalg.vector_norm(z_i-z_j, dim=-1)-delta).square().mean()
        optimizer_step(loss, optimizer, encoder.parameters())
        soft_update(target, encoder, 1-config.encoder_ema)
        if step % config.log_every == 0 or step == steps:
            entry = dict(step=step, loss=float(loss.detach()), mean_target=float(delta.mean()))
            history.append(entry); log("representation", **entry)
    freeze(encoder)
    with torch.no_grad():
        latents = torch.cat([encoder(*table.batch(np.arange(i, min(i+512, len(table.users))), device))
                             for i in range(0, len(table.users), 512)])
    if not torch.isfinite(latents).all():
        raise FloatingPointError("Non-finite frozen representation")
    return encoder, latents.detach(), history
