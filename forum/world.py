"""Eqs. (9)-(12): independent feedback, termination and latent models."""
import math
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .diffusion import DisplacementDiffusion
from .networks import mlp
from .utils import freeze, log, optimizer_step, training_batches


class WorldModel(nn.Module):
    def __init__(self, vectors, config, feedback_values=None, feedback_embedding=None):
        super().__init__()
        self.register_buffer("item_vectors", vectors.detach().clone())
        values = [] if feedback_values is None else feedback_values
        if feedback_values is not None and (not len(values) or len(set(values)) != len(values)
                                            or not np.isfinite(values).all()):
            raise ValueError("Feedback categories must be finite, nonempty and unique")
        self.register_buffer("feedback_values", torch.as_tensor(values, dtype=torch.float32, device=vectors.device))
        self.categorical = len(values) > 0
        context = config.latent_dim + vectors.shape[1]
        self.feedback_head = mlp(context, len(values) if self.categorical else 2, config.hidden_dim)
        self.terminal_head = mlp(context+1, 1, config.hidden_dim)
        self.feedback_embedding = (freeze(deepcopy(feedback_embedding)) if feedback_embedding is not None
                                   else nn.Sequential(nn.Linear(1, config.embedding_dim), nn.Tanh()))
        self.diffusion = DisplacementDiffusion(config.latent_dim, context+config.embedding_dim,
            config.hidden_dim, config.diffusion_steps)

    def context(self, z, action):
        return torch.cat([z, self.item_vectors[action]], -1)

    def feedback_distribution(self, context):
        params = self.feedback_head(context)
        if self.categorical:
            return torch.distributions.Categorical(logits=params)
        mean, log_scale = params.unbind(-1)
        return torch.distributions.Normal(mean, log_scale.clamp(-5, 2).exp())

    def losses(self, z, action, feedback, done, next_z, terminal_observed=None, dynamics_batch=None):
        context = self.context(z, action); dist = self.feedback_distribution(context)
        if self.categorical:
            matches = feedback[:, None] == self.feedback_values[None]
            if not matches.any(1).all():
                raise ValueError("Feedback outside configured categorical vocabulary")
            feedback_loss = -dist.log_prob(matches.long().argmax(-1)).mean()
        else:
            feedback_loss = -dist.log_prob(feedback).mean()
        observed = torch.ones_like(done) if terminal_observed is None else terminal_observed
        if observed.any():
            terminal_loss = F.binary_cross_entropy_with_logits(
                self.terminal_head(torch.cat([context[observed], feedback[observed, None]], -1)).squeeze(-1),
                done[observed].float())
        else:
            terminal_loss = feedback_loss * 0
        if dynamics_batch is None:
            keep = ~done & observed
            dz, da, dy, dn = z[keep], action[keep], feedback[keep], next_z[keep]
        else:
            dz, da, dy, dn = dynamics_batch
        if len(dz):
            dyn_context = torch.cat([self.context(dz, da), self.feedback_embedding(dy[:, None])], -1)
            dynamics_loss = self.diffusion.loss((dn-dz).detach(), dyn_context)
        else:
            dynamics_loss = feedback_loss * 0
        return dynamics_loss, feedback_loss, terminal_loss

    @torch.no_grad()
    def simulate(self, z, action):
        context = self.context(z, action); sampled = self.feedback_distribution(context).sample()
        feedback = self.feedback_values[sampled] if self.categorical else sampled
        terminal_probability = self.terminal_head(torch.cat([context, feedback[:, None]], -1)).squeeze(-1).sigmoid()
        done = torch.rand_like(terminal_probability) < terminal_probability
        next_z = torch.zeros_like(z)  # Absorbing terminal state.
        keep = ~done
        if keep.any():
            dyn_context = torch.cat([context[keep], self.feedback_embedding(feedback[keep, None])], -1)
            next_z[keep] = z[keep]+self.diffusion.sample(dyn_context)
        return feedback, done, next_z


def train_ensemble(table, latents, encoder, config, device, feedback_values):
    state = torch.as_tensor(table.state, device=device); nxt = torch.as_tensor(table.next_state, device=device)
    actions = torch.as_tensor(table.item, device=device)
    feedback = torch.as_tensor(table.reward, device=device); done = torch.as_tensor(table.done, device=device)
    observed = torch.as_tensor(table.terminal_observed, device=device)
    if not ((~done) & observed).any():
        raise ValueError("No nonterminal observations to fit dynamics")
    vectors = encoder.item_embedding.weight[1:].detach()
    models, history, bootstraps = [], [], []
    for member in range(config.ensemble_size):
        torch.manual_seed(config.seed+1000+member)
        rng = np.random.default_rng(config.seed+2000+member)
        rows, sessions = table.bootstrap_rows(rng); bootstraps.append(sessions)
        nt_rows = rows[table.terminal_observed[rows] & ~table.done[rows]]
        if not len(nt_rows):
            raise ValueError(f"Bootstrap member {member} has no observed nonterminal transitions")
        model = WorldModel(vectors, config, feedback_values, encoder.feedback_embedding).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        steps = config.world_steps or config.world_epochs*math.ceil(len(rows)/config.batch_size)
        batches = training_batches(rows, rng, config.batch_size, config.world_steps, config.world_epochs)
        for step, indices in enumerate(batches, 1):
            idx = torch.as_tensor(indices, device=device)
            # Eq. (11) is an expectation over D_nt, independently of the response batch.
            nt = torch.as_tensor(rng.choice(nt_rows, len(indices)), device=device)
            dynamics_batch = (latents[state[nt]], actions[nt], feedback[nt], latents[nxt[nt]])
            dyn, y, d = model.losses(latents[state[idx]], actions[idx], feedback[idx], done[idx],
                                    latents[nxt[idx]], observed[idx], dynamics_batch)
            loss = dyn+config.feedback_loss_weight*y+config.terminal_loss_weight*d
            optimizer_step(loss, optimizer, model.parameters())
            if step % config.log_every == 0 or step == steps:
                entry = dict(member=member, step=step, diffusion=float(dyn.detach()),
                             feedback=float(y.detach()), termination=float(d.detach()))
                history.append(entry); log("world", **entry)
        models.append(freeze(model))
    return models, history, bootstraps
