"""Deployment path: encoder + behavior support + actor. No world model or critic."""
import numpy as np
import torch

from .config import Config
from .networks import HistoryEncoder, ItemScorer
from .support import support_mask
from .utils import freeze


class Recommender:
    def __init__(self, checkpoint, device="cpu"):
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        self.config = Config(**saved["config"]).validate()
        self.device = torch.device(device)
        self.n_users, self.n_items = saved["n_users"], saved["n_items"]
        self.encoder = HistoryEncoder(self.n_users, self.n_items, self.config).to(device)
        self.encoder.load_state_dict(saved["encoder"])
        vectors = self.encoder.item_embedding.weight[1:].detach()
        self.behavior = ItemScorer(vectors, self.config.latent_dim, self.config.hidden_dim).to(device)
        self.actor = ItemScorer(vectors, self.config.latent_dim, self.config.hidden_dim).to(device)
        self.behavior.load_state_dict(saved["behavior"]); self.actor.load_state_dict(saved["actor"])
        for model in (self.encoder, self.behavior, self.actor):
            freeze(model)

    @torch.no_grad()
    def rank(self, users, histories, count=1):
        if len(users) != len(histories) or not users:
            raise ValueError("Users and histories must be nonempty and aligned")
        if count < 1 or any(u < 0 or u >= self.n_users for u in users):
            raise ValueError("Invalid requested count or unknown user")
        n, w = len(users), self.config.history_window
        actions = np.zeros((n, w), np.int64); feedback = np.zeros((n, w), np.float32)
        lengths = np.zeros(n, np.int64); seen = np.zeros((n, self.n_items), bool)
        for i, history in enumerate(histories):
            for a, y in history:
                if not isinstance(a, (int, np.integer)) or not 0 <= a < self.n_items or not np.isfinite(y):
                    raise ValueError("Invalid observed history")
                if self.config.no_repeat:
                    seen[i, a] = True
            recent = history[-w:]; lengths[i] = len(recent)
            for j, (a, y) in enumerate(recent):
                actions[i, j] = a+1; feedback[i, j] = y
        tensors = [torch.as_tensor(x, device=self.device) for x in (users, actions, feedback, lengths)]
        z = self.encoder(*tensors)
        support = support_mask(self.behavior(z), torch.as_tensor(seen, device=self.device),
                               self.config.support_k, self.config.support_threshold)
        scores = self.actor(z).masked_fill(~support, -torch.inf)
        result = []
        for i in range(n):
            k = min(count, int(support[i].sum()))
            result.append(scores[i].topk(k).indices.cpu().tolist())
        return result
