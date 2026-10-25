import torch
from torch import nn


def mlp(inputs, outputs, hidden, depth=2):
    layers = []
    for _ in range(depth):
        layers.extend([nn.Linear(inputs, hidden), nn.SiLU()]); inputs = hidden
    return nn.Sequential(*layers, nn.Linear(inputs, outputs))


class HistoryEncoder(nn.Module):
    """Eq. (5): user-conditioned GRU over strictly preceding observations."""
    def __init__(self, users, items, config):
        super().__init__()
        e, d = config.embedding_dim, config.latent_dim
        self.item_embedding = nn.Embedding(items + 1, e, padding_idx=0)
        self.user_embedding = nn.Embedding(users, d)
        self.feedback_embedding = nn.Sequential(nn.Linear(1, e), nn.Tanh())
        self.gru = nn.GRU(2 * e, d, batch_first=True)
        self.output = nn.Linear(d, d)

    def forward(self, users, actions, feedback, lengths):
        initial = self.user_embedding(users)
        tokens = torch.cat([self.item_embedding(actions), self.feedback_embedding(feedback[..., None])], -1)
        states, _ = self.gru(tokens, initial[None])
        last = states[torch.arange(len(users), device=users.device), (lengths-1).clamp_min(0)]
        last = torch.where((lengths > 0)[:, None], last, initial)
        return self.output(last)


class ItemScorer(nn.Module):
    """Two-layer MLP score s(z,e_a), evaluated over the fixed catalog."""
    def __init__(self, vectors, latent, hidden):
        super().__init__()
        self.register_buffer("item_vectors", vectors.detach().clone())
        self.network = mlp(latent + vectors.shape[1], 1, hidden)

    def forward(self, z):
        batch, items = len(z), len(self.item_vectors)
        x = torch.cat([z[:, None].expand(batch, items, -1), self.item_vectors[None].expand(batch, items, -1)], -1)
        return self.network(x).squeeze(-1)


class ValueNetwork(nn.Module):
    def __init__(self, latent, hidden):
        super().__init__()
        self.network = mlp(latent, 1, hidden)

    def forward(self, z):
        return self.network(z).squeeze(-1)
