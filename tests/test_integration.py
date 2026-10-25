from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from forum.checkpoint import load_training_bundle
from forum.config import Config
from forum.data import synthetic_sessions
from forum.serving import Recommender
from forum.train import run_training


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temp.name)/"run"
        c = Config(history_window=5, latent_dim=4, embedding_dim=4, hidden_dim=8,
                   ensemble_size=2, diffusion_steps=2, uncertainty_samples=2,
                   representation_steps=2, world_steps=2, behavior_steps=2, value_steps=2,
                   policy_steps=2, rollout_batch=4, batch_size=8, support_k=10, rollout_horizon=2)
        data, provenance = synthetic_sessions(users=4)
        with redirect_stdout(io.StringIO()):
            cls.summary = run_training(data, provenance, c, cls.out, n_users=4, n_items=40,
                                       feedback_values=[1., 2., 3., 4., 5.])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_training_and_frozen_invariants(self):
        self.assertEqual(self.summary["status"], "passed")
        self.assertTrue(all(self.summary["checks"].values()))

    def test_independent_world_bootstraps_saved(self):
        chosen = json.loads((self.out/"bootstrap_sessions.json").read_text())
        self.assertEqual(len(chosen), 2)
        self.assertNotEqual(chosen[0], chosen[1])

    def test_training_checkpoint_roundtrip(self):
        saved = torch.load(self.out/"training.pt", weights_only=True)
        models = load_training_bundle(self.out/"training.pt")
        self.assertEqual(len(models["worlds"]), 2)
        for loaded, original in zip(models["worlds"], saved["worlds"]):
            for key, value in loaded.state_dict().items():
                self.assertTrue(torch.equal(value, original[key]))
        self.assertTrue(all(not p.requires_grad for m in models["worlds"] for p in m.parameters()))

    def test_serving_30_turns_without_world_calls(self):
        histories = [[] for _ in range(4)]
        with patch("forum.world.WorldModel.simulate", side_effect=AssertionError("Serving called world")):
            model = Recommender(self.out/"serving.pt")
            for _ in range(30):
                choices = model.rank([0, 1, 2, 3], histories)
                for h, a in zip(histories, choices):
                    h.append((a[0], 3.))
        self.assertTrue(all(len({a for a, _ in h}) == 30 for h in histories))

    def test_rollout_member_and_terminal_consistency(self):
        with np.load(self.out/"last_rollouts.npz") as b:
            for trajectory in np.unique(b["trajectory"]):
                selected = b["trajectory"] == trajectory
                self.assertEqual(len(np.unique(b["member"][selected])), 1)
                done = b["done"][selected]
                self.assertFalse(done[:-1].any())

    def test_cannot_overwrite_previous_run(self):
        data, provenance = synthetic_sessions(users=4)
        with self.assertRaises(FileExistsError):
            run_training(data, provenance, Config(), self.out)

    def test_epoch_budget_path_runs_all_four_stages(self):
        data, provenance = synthetic_sessions(users=2)
        config = Config(history_window=3, latent_dim=4, embedding_dim=4, hidden_dim=8,
            ensemble_size=2, diffusion_steps=2, uncertainty_samples=2, representation_epochs=1,
            world_epochs=1, representation_steps=0, world_steps=0, behavior_steps=2,
            value_steps=2, policy_steps=2, rollout_batch=4, batch_size=16, log_every=1)
        output = Path(self.temp.name)/'epochs'
        with redirect_stdout(io.StringIO()):
            result = run_training(data, provenance, config, output, n_users=2, n_items=40,
                                  feedback_values=[1., 2., 3., 4., 5.])
        self.assertEqual(result['status'], 'passed')
        history = json.loads((output/'training_history.json').read_text())
        self.assertEqual(len(history['representation']), (result['rows']+15)//16)
        self.assertTrue(all(history[key] for key in ('world', 'policy')))


if __name__ == "__main__":
    unittest.main()
