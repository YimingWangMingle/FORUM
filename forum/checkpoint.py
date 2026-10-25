"""Reload all frozen FORUM components for model diagnostics or further research."""
import torch
from .config import Config
from .networks import HistoryEncoder, ItemScorer, ValueNetwork
from .world import WorldModel
from .utils import freeze


def load_training_bundle(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    config = Config(**saved["config"]).validate()
    encoder = HistoryEncoder(saved["n_users"], saved["n_items"], config).to(device)
    encoder.load_state_dict(saved["encoder"]); freeze(encoder)
    vectors = encoder.item_embedding.weight[1:].detach()
    result = dict(config=config, encoder=encoder, worlds=[])
    for name in ("behavior", "actor"):
        model = ItemScorer(vectors, config.latent_dim, config.hidden_dim).to(device)
        model.load_state_dict(saved[name]); result[name] = freeze(model)
    for name in ("raw", "penalized", "target_raw", "target_penalized"):
        model = ValueNetwork(config.latent_dim, config.hidden_dim).to(device)
        model.load_state_dict(saved[name]); result[name] = freeze(model)
    for state in saved["worlds"]:
        if "diffusion.mean" in state:
            raise ValueError("Legacy normalized-diffusion checkpoint; retrain with the audited implementation")
        model = WorldModel(vectors, config, saved["feedback_values"], encoder.feedback_embedding).to(device)
        model.load_state_dict(state); result["worlds"].append(freeze(model))
    return result
