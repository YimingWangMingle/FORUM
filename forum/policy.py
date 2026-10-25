"""Support-constrained A2C with separate raw and penalized critics, Eqs. (17)-(20)."""
from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F

from .networks import ItemScorer, ValueNetwork
from .support import support_mask
from .uncertainty import estimate_uncertainty
from .utils import freeze, log, optimizer_step, soft_update


def action_distribution(actor, behavior, z, seen, config):
    with torch.no_grad():
        mask = support_mask(behavior(z), seen, config.support_k, config.support_threshold)
    return torch.distributions.Categorical(logits=actor(z).masked_fill(~mask, -torch.inf)), mask


def td_targets(reward, penalty, done, next_raw, next_penalized, gamma):
    # A rollout time limit is not a termination. No truncation multiplier here.
    raw_tail = torch.where(done, torch.zeros_like(next_raw), next_raw)
    penalized_tail = torch.where(done, torch.zeros_like(next_penalized), next_penalized)
    return (reward+gamma*raw_tail).detach(), (reward-penalty+gamma*penalized_tail).detach()


def actor_critic_losses(raw_pred, penalized_pred, log_probability, raw_target, penalized_target):
    """Eq. (20): only the actor receives policy-gradient derivatives."""
    raw_loss = F.mse_loss(raw_pred, raw_target.detach())
    penalized_loss = F.mse_loss(penalized_pred, penalized_target.detach())
    actor_loss = -(log_probability*(penalized_target-penalized_pred).detach()).mean()
    return raw_loss, penalized_loss, actor_loss


@torch.no_grad()
def collect_rollouts(table, latents, models, actor, behavior, target_raw, config, rng):
    device = latents.device
    eligible = np.arange(len(table))
    if config.no_repeat:
        eligible = eligible[(~table.seen[table.state]).any(-1).numpy()]
    roots = rng.choice(eligible, config.rollout_batch)
    z = latents[torch.as_tensor(table.state[roots], device=device)].clone()
    seen = table.seen[table.state[roots]].to(device).clone()
    if not config.no_repeat:
        seen.zero_()
    members = torch.as_tensor(rng.integers(len(models), size=len(roots)), device=device)
    alive = torch.ones(len(roots), dtype=torch.bool, device=device)
    storage = {k: [] for k in ("z", "action", "reward", "penalty", "done", "next_z", "seen", "member", "trajectory", "turn", "u_reward", "u_value", "model_done", "catalog_exhausted")}
    for turn in range(config.rollout_horizon):
        idx = torch.where(alive & (~seen).any(-1))[0]
        if not len(idx):
            break
        current, current_seen = z[idx], seen[idx].clone()
        dist, mask = action_distribution(actor, behavior, current, current_seen, config)
        action = dist.sample()
        if not mask.gather(1, action[:, None]).all():
            raise AssertionError("Unsupported simulated action")
        exhausted = (current_seen.sum(-1)+1 == table.n_items) if config.no_repeat else torch.zeros_like(action, dtype=torch.bool)
        penalty, ur, uv = estimate_uncertainty(models, target_raw, current, action, config, ~exhausted)
        reward = torch.empty(len(idx), device=device); done = torch.empty(len(idx), device=device, dtype=torch.bool)
        nxt = torch.empty_like(current)
        for member, model in enumerate(models):
            use = members[idx] == member
            if use.any():
                reward[use], done[use], nxt[use] = model.simulate(current[use], action[use])
        model_done = done.clone()
        done |= exhausted
        nxt[done] = 0
        values = (current, action, reward, penalty, done, nxt, current_seen, members[idx], idx,
                  torch.full_like(idx, turn), ur, uv, model_done, exhausted)
        for key, value in zip(storage, values):
            storage[key].append(value.clone())
        z[idx] = nxt; alive[idx] = ~done
        if config.no_repeat:
            seen[idx, action] = True
    if not storage["z"]:
        raise ValueError("No simulated transitions")
    return {k: torch.cat(v).detach() for k, v in storage.items()}


def fit_policy(table, latents, encoder, models, config, device):
    rng = np.random.default_rng(config.seed+3000); torch.manual_seed(config.seed+3000)
    vectors = encoder.item_embedding.weight[1:].detach()
    states = torch.as_tensor(table.state, device=device); actions = torch.as_tensor(table.item, device=device)
    history = []
    raw = ValueNetwork(config.latent_dim, config.hidden_dim).to(device)
    target_raw = freeze(deepcopy(raw))
    raw_opt = torch.optim.Adam(raw.parameters(), lr=config.learning_rate)
    returns, discounts, ends = [torch.as_tensor(x, device=device) for x in table.logged_returns(config.gamma)]
    for step in range(1, config.value_steps+1):
        rows = torch.as_tensor(rng.integers(len(table), size=config.batch_size), device=device)
        with torch.no_grad():
            target = returns[rows]+discounts[rows]*target_raw(latents[ends[rows]])
        loss = F.mse_loss(raw(latents[states[rows]]), target)
        optimizer_step(loss, raw_opt, raw.parameters()); soft_update(target_raw, raw, config.target_tau)
        if step % config.log_every == 0 or step == config.value_steps:
            log("raw_value_init", step=step, loss=float(loss.detach()))
    # Algorithm 1: initialize the raw critic before fitting behavior support.
    behavior = ItemScorer(vectors, config.latent_dim, config.hidden_dim).to(device)
    opt = torch.optim.Adam(behavior.parameters(), lr=config.learning_rate)
    for step in range(1, config.behavior_steps+1):
        rows = torch.as_tensor(rng.integers(len(table), size=config.batch_size), device=device)
        loss = F.cross_entropy(behavior(latents[states[rows]]), actions[rows])
        optimizer_step(loss, opt, behavior.parameters())
        if step % config.log_every == 0 or step == config.behavior_steps:
            log("behavior", step=step, loss=float(loss.detach()))
    freeze(behavior)
    actor = deepcopy(behavior).requires_grad_(True).train()  # BC initialization, documented choice.
    penalized = deepcopy(raw); target_penalized = freeze(deepcopy(penalized))
    actor_opt = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    penalized_opt = torch.optim.Adam(penalized.parameters(), lr=config.learning_rate)
    # Use an exact initialized raw target; subsequent updates are soft, between updates only.
    target_raw.load_state_dict(raw.state_dict())
    total_simulated, total_terminal = 0, 0
    last_batch = None
    for step in range(1, config.policy_steps+1):
        batch = collect_rollouts(table, latents, models, actor, behavior, target_raw, config, rng)
        with torch.no_grad():
            next_raw = target_raw(batch["next_z"]); next_w = target_penalized(batch["next_z"])
            b_raw, b = td_targets(batch["reward"], batch["penalty"], batch["done"], next_raw, next_w, config.gamma)
        raw_pred = raw(batch["z"]); w_pred = penalized(batch["z"])
        dist, _ = action_distribution(actor, behavior, batch["z"], batch["seen"], config)
        raw_loss, w_loss, actor_loss = actor_critic_losses(
            raw_pred, w_pred, dist.log_prob(batch["action"]), b_raw, b)
        optimizer_step(actor_loss, actor_opt, actor.parameters())
        optimizer_step(raw_loss, raw_opt, raw.parameters())
        optimizer_step(w_loss, penalized_opt, penalized.parameters())
        soft_update(target_raw, raw, config.target_tau)
        soft_update(target_penalized, penalized, config.target_tau)
        total_simulated += len(batch["reward"]); total_terminal += int(batch["done"].sum())
        if step % config.log_every == 0 or step == config.policy_steps:
            entry = dict(step=step, actor_loss=float(actor_loss.detach()), raw_loss=float(raw_loss.detach()),
                penalized_loss=float(w_loss.detach()), mean_reward=float(batch["reward"].mean()),
                mean_penalty=float(batch["penalty"].mean()), reward_disagreement=float(batch["u_reward"].mean()),
                value_disagreement=float(batch["u_value"].mean()), transitions=len(batch["reward"]))
            history.append(entry); log("policy", **entry)
        last_batch = batch
    return dict(actor=freeze(actor), behavior=behavior, raw=freeze(raw), penalized=freeze(penalized),
                target_raw=target_raw, target_penalized=target_penalized, history=history,
                simulated=total_simulated, terminals=total_terminal, last_batch=last_batch)
