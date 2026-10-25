from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from forum.data import HistoryTable, Session, coat_sessions


def sessions():
    return [Session("terminal", 0, [0, 1, 2], [1., 2., 3.], True),
            Session("censored", 1, [0, 2, 1], [2., 3., 4.], False)]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.table = HistoryTable(sessions(), 2, n_items=5)

    def test_current_action_and_feedback_are_not_in_history(self):
        t = self.table
        self.assertEqual(t.lengths[t.state[0]], 0)
        self.assertEqual(t.actions[t.state[1]].tolist(), [1, 0])
        self.assertEqual(t.feedback[t.state[1]].tolist(), [1., 0.])
        changed = sessions(); changed[0].feedback[1:] = [99., 88.]
        other = HistoryTable(changed, 2, n_items=5)
        np.testing.assert_array_equal(t.feedback[t.state[1]], other.feedback[other.state[1]])

    def test_terminal_window_valid_with_zero_tail(self):
        total, length, alive, _ = self.table.window_target(1, 5, .5)
        self.assertEqual((total, length, alive), (3.5, 2, False))

    def test_censored_window_excluded(self):
        self.assertIsNone(self.table.window_target(4, 5, .5))

    def test_censored_boundary_has_history_but_unknown_survival(self):
        total, length, alive, successor = self.table.window_target(4, 2, .5)
        self.assertEqual((total, length, alive), (5., 2, False))
        self.assertEqual(successor, 7)

    def test_termination_on_last_step_disables_successor(self):
        self.assertFalse(self.table.window_target(0, 3, .9)[2])

    def test_action_matched_distinct_pairs(self):
        pairs = self.table.pairs([1, 2, 3, 5], .9).sample(np.random.default_rng(0), 100)
        self.assertTrue((pairs["left"] != pairs["right"]).all())
        np.testing.assert_array_equal(self.table.item[pairs["left"]], self.table.item[pairs["right"]])
        for row, k in zip(pairs["left"], pairs["horizon"]):
            self.assertIsNotNone(self.table.window_target(row, k, .9))

    def test_bootstrap_is_whole_sessions(self):
        rows, chosen = self.table.bootstrap_rows(np.random.default_rng(4))
        for session in range(2):
            count = chosen.count(session)
            for row in np.flatnonzero(self.table.session_index == session):
                self.assertEqual(np.count_nonzero(rows == row), count)

    def test_logged_return_bootstraps_censor_only(self):
        returns, discount, end = self.table.logged_returns(.5)
        self.assertEqual(discount[0], 0.)
        self.assertEqual(discount[3], .125)
        self.assertEqual(returns[3], 4.5)
        self.assertEqual(end[3], 7)

    def test_full_seen_set_kept_outside_history_window(self):
        self.assertEqual(self.table.actions[3].tolist(), [2, 3])
        self.assertEqual(self.table.seen[3].tolist(), [True, True, True, False, False])

    def test_duplicate_sessions_and_actions_rejected(self):
        with self.assertRaises(ValueError):
            HistoryTable([sessions()[0], sessions()[0]], 2)
        with self.assertRaises(ValueError):
            HistoryTable([Session("x", 0, [0, 0], [1., 2.], False)], 2)

    def test_coat_reads_only_explicit_training_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"train.ascii"
            matrix = np.zeros((290, 300), dtype=int); matrix[:, :24] = 3
            np.savetxt(p, matrix, fmt="%d")
            (Path(d)/"test.ascii").write_text("FORBIDDEN")
            original = np.loadtxt; reads = []
            def guarded(path, *args, **kwargs):
                self.assertEqual(Path(path), p); reads.append(str(path))
                return original(path, *args, **kwargs)
            with patch("numpy.loadtxt", side_effect=guarded):
                data, provenance = coat_sessions(p, max_users=2)
            self.assertEqual(len(reads), 1)
            self.assertTrue(all(not s.terminated for s in data))
            self.assertFalse(provenance["genuine_sequential_benchmark"])


if __name__ == "__main__":
    unittest.main()
