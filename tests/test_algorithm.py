import unittest
from unittest.mock import patch
import torch

from forum.config import Config
from forum.networks import HistoryEncoder
from forum.policy import td_targets
from forum.representation import behavioral_target
from forum.support import support_mask
from forum.uncertainty import disagreement, estimate_uncertainty
from forum.world import WorldModel


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0); torch.set_num_threads(1)
        self.config = Config(latent_dim=4, embedding_dim=4, hidden_dim=8,
                             ensemble_size=2, diffusion_steps=4, uncertainty_samples=2)

    def model(self):
        return WorldModel(torch.randn(6, 4), self.config, [1., 2., 3., 4., 5.])

    def test_representation_target_formula_and_stop_gradient(self):
        c = self.config
        nxt = torch.tensor([[3., 4.]], requires_grad=True)
        target = behavioral_target(torch.tensor([5.]), torch.tensor([2.]), torch.tensor([3.]),
            torch.tensor([1.]), torch.tensor([3.]), torch.tensor([1.]), nxt, torch.zeros_like(nxt), c)
        self.assertAlmostEqual(target.item(), 3 + .5*2/3 + .1*.99**3*5, places=5)
        self.assertFalse(target.requires_grad)

    def test_terminal_pair_masks_successor_distance(self):
        z = torch.ones(1, 4)*100
        target = behavioral_target(torch.tensor([1.]), torch.tensor([1.]), torch.tensor([1.]),
            torch.tensor([1.]), torch.tensor([3.]), torch.tensor([0.]), z, torch.zeros_like(z), self.config)
        self.assertEqual(target.item(), 0.)

    def test_encoder_padding_does_not_change_empty_history(self):
        m = HistoryEncoder(2, 6, self.config)
        users = torch.tensor([0]); lengths = torch.tensor([0])
        a = m(users, torch.zeros(1, 5, dtype=torch.long), torch.zeros(1, 5), lengths)
        b = m(users, torch.ones(1, 5, dtype=torch.long), torch.ones(1, 5)*9, lengths)
        torch.testing.assert_close(a, b)

    def test_support_topk_threshold_and_nx0(self):
        logits = torch.tensor([[4., 3., 2., 1.]])
        seen = torch.tensor([[True, False, False, False]])
        mask = support_mask(logits, seen, 2, .01)
        self.assertEqual(mask.tolist(), [[False, True, True, False]])

    def test_support_fallback_retains_best_legal_action(self):
        mask = support_mask(torch.tensor([[3., 2., 1.]]), torch.tensor([[True, False, False]]), 2, .99)
        self.assertEqual(mask.tolist(), [[False, True, False]])

    def test_exhausted_catalog_fails_explicitly(self):
        with self.assertRaises(ValueError):
            support_mask(torch.ones(1, 3), torch.ones(1, 3, dtype=torch.bool), 2, 0.)

    def test_disagreement_population_std(self):
        p, ur, uv = disagreement(torch.tensor([[1.], [3.]]), torch.tensor([[2.], [6.]]), .9, .1, .2)
        self.assertEqual(ur.item(), 1.)
        self.assertEqual(uv.item(), 2.)
        self.assertAlmostEqual(p.item(), .46, places=6)

    def test_average_within_member_before_disagreement(self):
        class Model:
            def simulate(self, z, a):
                return torch.tensor([1., 5.]), torch.tensor([False, False]), torch.tensor([[1.], [5.]])
        penalty, ur, uv = estimate_uncertainty([Model(), Model()], lambda z: z[:, 0],
                                               torch.zeros(1, 1), torch.zeros(1, dtype=torch.long), self.config)
        self.assertEqual((penalty.item(), ur.item(), uv.item()), (0., 0., 0.))

    def test_td_raw_critic_never_receives_penalty(self):
        reward = torch.tensor([3., 3.]); penalty = torch.tensor([1., 1.]); done = torch.tensor([False, True])
        raw, penalized = td_targets(reward, penalty, done, torch.tensor([10., 10.]), torch.tensor([20., 20.]), .9)
        torch.testing.assert_close(raw, torch.tensor([12., 3.]))
        torch.testing.assert_close(penalized, torch.tensor([20., 2.]))

    def test_nonterminal_horizon_retains_bootstrap(self):
        raw, _ = td_targets(torch.tensor([2.]), torch.tensor([0.]), torch.tensor([False]),
                            torch.tensor([5.]), torch.tensor([5.]), .9)
        self.assertEqual(raw.item(), 6.5)

    def test_diffusion_skips_terminal_training_rows(self):
        m = self.model()
        with patch.object(m.diffusion, "loss", side_effect=AssertionError("must skip")):
            losses = m.losses(torch.zeros(2, 4), torch.tensor([0, 1]), torch.tensor([1., 2.]),
                              torch.ones(2, dtype=torch.bool), torch.zeros(2, 4))
        self.assertEqual(losses[0].item(), 0.)

    def test_world_model_all_losses_backpropagate(self):
        m = self.model()
        loss = sum(m.losses(torch.randn(4, 4), torch.tensor([0, 1, 2, 3]), torch.tensor([1., 2., 3., 4.]),
                            torch.tensor([True, False, False, False]), torch.randn(4, 4)))
        loss.backward()
        for sub in (m.feedback_head, m.terminal_head, m.diffusion.denoiser):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in sub.parameters()))

    def test_terminal_simulation_never_calls_diffusion(self):
        m = self.model()
        with torch.no_grad():
            for p in m.terminal_head.parameters():
                p.zero_()
            m.terminal_head[-1].bias.fill_(100)
        with patch.object(m.diffusion, "sample", side_effect=AssertionError("must skip")):
            y, d, nxt = m.simulate(torch.zeros(8, 4), torch.zeros(8, dtype=torch.long))
        self.assertTrue(d.all()); self.assertTrue((nxt == 0).all())
        self.assertTrue(torch.isin(y, torch.arange(1., 6.)).all())

    def test_gaussian_feedback_path(self):
        m = WorldModel(torch.randn(6, 4), self.config)
        losses = m.losses(torch.zeros(2, 4), torch.tensor([0, 1]), torch.tensor([.7, 1.2]),
                          torch.tensor([False, True]), torch.ones(2, 4))
        self.assertTrue(all(torch.isfinite(x) for x in losses))

    def test_config_rejects_single_member(self):
        with self.assertRaises(ValueError):
            Config(ensemble_size=1).validate()


if __name__ == "__main__":
    unittest.main()
