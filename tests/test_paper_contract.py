"""Independent numerical and data-boundary checks for the manuscript contract."""
import unittest
from unittest.mock import patch
import numpy as np
import torch

from forum.config import Config
from forum.data import HistoryTable, Session
from forum.diffusion import DisplacementDiffusion
from forum.policy import actor_critic_losses, collect_rollouts, td_targets
from forum.uncertainty import estimate_uncertainty
from forum.utils import training_batches
from forum.world import WorldModel


class PaperContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)
        self.c = Config(latent_dim=2, embedding_dim=2, hidden_dim=4,
                        ensemble_size=2, diffusion_steps=4, uncertainty_samples=2)

    def test_raw_displacement_forward_noise_and_squared_vector_loss(self):
        model = DisplacementDiffusion(2, 3, 4, 4)
        displacement = torch.tensor([[2., 8.], [4., 6.]], requires_grad=True)
        noise = torch.tensor([[1., 2.], [3., 4.]])
        times = torch.tensor([0, 2])
        with patch('torch.randint', return_value=times), patch('torch.randn_like', return_value=noise), \
                patch.object(model, 'predict', return_value=torch.zeros_like(noise)) as predict:
            loss = model.loss(displacement, torch.zeros(2, 3))
        abar = model.cumulative[times, None]
        expected_noisy = abar.sqrt()*displacement.detach()+(1-abar).sqrt()*noise
        torch.testing.assert_close(predict.call_args.args[0], expected_noisy)
        self.assertFalse(predict.call_args.args[0].requires_grad)
        self.assertEqual(loss.item(), 15.)  # ((1+4)+(9+16))/2, not 7.5.

    def test_ddpm_reverse_uses_all_steps_and_raw_latent_units(self):
        model = DisplacementDiffusion(2, 3, 4, 4)
        initial = torch.tensor([[2., -3.]])
        with patch('torch.randn', return_value=initial.clone()), \
                patch('torch.randn_like', side_effect=lambda x: torch.zeros_like(x)), \
                patch.object(model, 'predict', side_effect=lambda x, c, t: torch.zeros_like(x)) as predict:
            sampled = model.sample(torch.zeros(1, 3))
        torch.testing.assert_close(sampled, initial/model.alpha.prod().sqrt())
        self.assertEqual([x.args[2].item() for x in predict.call_args_list], [3, 2, 1, 0])
        self.assertFalse(sampled.requires_grad)

    def test_pair_then_horizon_sampling_does_not_favor_long_pairs(self):
        table = HistoryTable([
            Session('short', 0, [0], [1.], False),
            Session('long', 1, [0, 1, 2], [2., 3., 4.], False),
            Session('terminal', 2, [0, 3, 4], [3., 4., 5.], True)], 3)
        sample = table.pairs((1, 3), .9).sample(np.random.default_rng(7), 12000)
        pairs = np.stack([sample['left'], sample['right']], -1)
        _, counts = np.unique(pairs, axis=0, return_counts=True)
        self.assertEqual(len(counts), 6)
        self.assertTrue((np.abs(counts/len(pairs)-1/6) < .02).all())
        both_long = (pairs != 0).all(-1)
        self.assertAlmostEqual((sample['horizon'][both_long] == 3).mean(), .5, delta=.03)
        self.assertTrue((sample['horizon'][~both_long] == 1).all())

    def test_unknown_terminal_is_not_a_negative_label_or_dynamics_example(self):
        model = WorldModel(torch.randn(3, 2), self.c, [1., 2., 3.])
        z = torch.zeros(2, 2)
        with patch.object(model.diffusion, 'loss', side_effect=AssertionError('Unknown boundary used')):
            dyn, feedback, terminal = model.losses(z, torch.tensor([0, 1]), torch.tensor([1., 2.]),
                torch.zeros(2, dtype=torch.bool), z+1, torch.zeros(2, dtype=torch.bool))
        self.assertEqual(dyn.item(), 0.)
        self.assertEqual(terminal.item(), 0.)
        (dyn+feedback+terminal).backward()
        self.assertTrue(all(p.grad is None for p in model.terminal_head.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.feedback_head.parameters()))

    def test_diffusion_expectation_can_use_nonterminal_pool_independent_of_response_batch(self):
        model = WorldModel(torch.randn(3, 2), self.c, [1., 2., 3.])
        z = torch.zeros(2, 2)
        dynamics = (z+2, torch.tensor([1, 2]), torch.tensor([2., 3.]), z+5)
        with patch.object(model.diffusion, 'loss', return_value=torch.tensor(7.)) as loss:
            dyn, _, _ = model.losses(z, torch.tensor([0, 1]), torch.tensor([1., 2.]),
                torch.ones(2, dtype=torch.bool), z, dynamics_batch=dynamics)
        self.assertEqual(dyn.item(), 7.)
        torch.testing.assert_close(loss.call_args.args[0], torch.full((2, 2), 3.))

    def test_world_simulation_conditions_on_generated_feedback(self):
        model = WorldModel(torch.randn(3, 2), self.c, [1., 2., 3.])
        with torch.no_grad():
            for p in model.terminal_head.parameters():
                p.zero_()
            model.terminal_head[-1].bias.fill_(-100)
        z = torch.zeros(2, 2)
        with patch.object(model.diffusion, 'sample', return_value=torch.ones_like(z)) as sample:
            y, done, nxt = model.simulate(z, torch.tensor([0, 1]))
        torch.testing.assert_close(sample.call_args.args[0][:, -2:], model.feedback_embedding(y[:, None]))
        torch.testing.assert_close(nxt, z+1)
        self.assertFalse(done.any())

    def test_world_epochs_visit_every_bootstrapped_row_with_multiplicity(self):
        rows = np.array([0, 1, 1, 2, 3])
        batches = list(training_batches(rows, np.random.default_rng(4), 2, 0, 2))
        self.assertEqual([len(x) for x in batches], [2, 2, 1, 2, 2, 1])
        for epoch in range(2):
            np.testing.assert_array_equal(np.sort(np.concatenate(batches[epoch*3:epoch*3+3])), rows)

    def test_actor_and_critic_losses_have_exact_values_and_gradient_boundaries(self):
        raw = torch.tensor([1., 2.], requires_grad=True)
        penalized = torch.tensor([2., 4.], requires_grad=True)
        logp = torch.tensor([-.2, -.4], requires_grad=True)
        raw_target = torch.tensor([4., 6.], requires_grad=True)
        target = torch.tensor([5., 2.], requires_grad=True)
        lr, lw, la = actor_critic_losses(raw, penalized, logp, raw_target, target)
        self.assertEqual(lr.item(), 12.5)
        self.assertEqual(lw.item(), 6.5)
        self.assertAlmostEqual(la.item(), -.1, places=6)
        la.backward()
        torch.testing.assert_close(logp.grad, torch.tensor([-1.5, 1.]))
        self.assertIsNone(penalized.grad)
        self.assertIsNone(target.grad)
        (lr+lw).backward()
        self.assertIsNone(raw_target.grad)
        self.assertIsNone(target.grad)

    def test_terminal_bootstrap_masks_even_nonfinite_unused_value(self):
        raw, value = td_targets(torch.tensor([2.]), torch.tensor([.5]), torch.tensor([True]),
                               torch.tensor([float('nan')]), torch.tensor([float('inf')]), .9)
        self.assertEqual(raw.item(), 2.)
        self.assertEqual(value.item(), 1.5)

    def rollout(self, items, horizon):
        table = HistoryTable([Session('a', 0, [0], [1.], False)], 1, n_items=items)
        class Model:
            def simulate(self, z, a):
                return torch.ones(len(z)), torch.zeros(len(z), dtype=torch.bool), z+1
        c = Config(latent_dim=2, rollout_batch=3, rollout_horizon=horizon, uncertainty_samples=2)
        scorer = lambda z: torch.zeros(len(z), items)
        return collect_rollouts(table, torch.zeros(2, 2), [Model(), Model()], scorer, scorer,
                                lambda z: z.sum(-1)+5, c, np.random.default_rng(4))

    def test_catalog_exhaustion_zeros_risk_continuation_and_td_tail(self):
        batch = self.rollout(1, 3)
        self.assertEqual(len(batch['reward']), 3)
        self.assertTrue(batch['catalog_exhausted'].all())
        self.assertFalse(batch['model_done'].any())
        self.assertTrue(batch['done'].all())
        self.assertTrue((batch['next_z'] == 0).all())
        self.assertTrue((batch['u_value'] == 0).all())
        raw, _ = td_targets(batch['reward'], batch['penalty'], batch['done'],
                            torch.full((3,), 99.), torch.full((3,), 99.), .9)
        torch.testing.assert_close(raw, batch['reward'])

    def test_horizon_truncation_is_not_catalog_or_model_termination(self):
        batch = self.rollout(3, 1)
        self.assertFalse(batch['done'].any())
        self.assertFalse(batch['catalog_exhausted'].any())
        raw, _ = td_targets(batch['reward'], batch['penalty'], batch['done'],
                            torch.full((3,), 5.), torch.full((3,), 5.), .9)
        torch.testing.assert_close(raw, batch['reward']+4.5)

    def test_no_critic_calls_for_absorbing_uncertainty_samples(self):
        class Model:
            def simulate(self, z, a):
                return torch.ones(len(z)), torch.ones(len(z), dtype=torch.bool), z
        with patch('forum.networks.ValueNetwork.forward', side_effect=AssertionError('Unused')):
            from forum.networks import ValueNetwork
            _, _, uv = estimate_uncertainty([Model(), Model()], ValueNetwork(2, 4),
                torch.zeros(1, 2), torch.zeros(1, dtype=torch.long), self.c)
        self.assertEqual(uv.item(), 0.)

    def test_data_records_unknown_labels_separately(self):
        table = HistoryTable([Session('c', 0, [0, 1], [1., 2.], False),
                              Session('t', 1, [0, 1], [1., 2.], True)], 2, n_items=3)
        self.assertEqual(table.terminal_observed.tolist(), [True, False, True, True])
        self.assertEqual(table.done.tolist(), [False, False, False, True])
        self.assertFalse(table.window_target(0, 2, .9)[2])
        self.assertTrue(table.window_target(0, 1, .9)[2])

    def test_configuration_rejects_nonfinite_and_fractional_counts(self):
        for kwargs in ({'lambda_value': float('nan')}, {'learning_rate': float('inf')},
                       {'world_epochs': 1.5}, {'horizons': (1, 2.5)}, {'no_repeat': 'false'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs).validate()


if __name__ == '__main__':
    unittest.main()
