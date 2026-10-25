import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from forum.evaluate import COAT_GT_SHA256, evaluate_coat


class FakeRecommender:
    n_users = 290
    n_items = 300
    class config:
        no_repeat = True
    def __init__(self, *args):
        pass
    def rank(self, users, histories):
        return [[len(h)] for h in histories]


class EvaluationTests(unittest.TestCase):
    def test_protocol_and_independent_reward_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"summary.json").write_text(json.dumps(dict(status="passed", checkpoints={"serving.pt":"modelhash"})))
            (root/"input.json").write_text('{}')
            matrix = root/"fixture_matrix"; np.savetxt(matrix, np.full((290, 300), 3), fmt="%d")
            def digest(path):
                return COAT_GT_SHA256 if Path(path) == matrix else "modelhash"
            with patch("forum.evaluate.sha256", side_effect=digest), patch("forum.evaluate.Recommender", FakeRecommender):
                result = evaluate_coat(root, matrix, root/"evaluation.json")
            self.assertEqual(result["mean_return"], 90.)
            self.assertEqual(result["protocol"], dict(N=4, M=0, T=30, NX_0=True, evaluation_users="all 290 exactly once"))
            records = json.loads((root/"evaluation.json").read_text())["users"]
            self.assertEqual(len({r["user"] for r in records}), 290)
            self.assertTrue(all(sum(r["rewards"]) == r["return_"] and len(set(r["actions"])) == 30 for r in records))

    def test_incomplete_training_rejected_before_matrix_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"summary.json").write_text('{"status":"failed"}')
            with patch("numpy.loadtxt", side_effect=AssertionError("Must not read matrix")):
                with self.assertRaises(ValueError):
                    evaluate_coat(root, root/"missing", root/"evaluation.json")
