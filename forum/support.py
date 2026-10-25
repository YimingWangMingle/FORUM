"""Eq. (17) and optional NX_0 legality, shared by training and deployment."""
import torch


def support_mask(behavior_logits, seen, top_k, threshold):
    legal = ~seen
    if not legal.any(-1).all():
        raise ValueError("Catalog exhausted; no legal action")
    probabilities = behavior_logits.softmax(-1)
    ranked = probabilities.masked_fill(~legal, -1)
    indices = ranked.topk(min(top_k, ranked.shape[-1]), dim=-1).indices
    selected = torch.zeros_like(legal).scatter_(1, indices, True)
    selected &= legal & (probabilities >= threshold)
    missing = ~selected.any(-1)
    if missing.any():
        selected[missing, ranked[missing].argmax(-1)] = True
    return selected
