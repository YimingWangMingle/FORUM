"""Exercise the distributable script and the public JSONL train/recommend CLI."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = cls.root/"toy.jsonl"
        cls.run_dir = cls.root/"model"
        subprocess.run([sys.executable, str(ROOT/"scripts/make_toy_data.py"), "--out", str(cls.data)],
                       cwd=ROOT, capture_output=True, text=True, check=True)
        subprocess.run([sys.executable, "-m", "forum", "train", "--sessions", str(cls.data),
                        "--n-users", "16", "--n-items", "40", "--feedback-values", "1,2,3,4,5",
                        "--config", str(ROOT/"configs/smoke.json"), "--out", str(cls.run_dir)],
                       cwd=ROOT, capture_output=True, text=True, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_jsonl_training_completes(self):
        summary = json.loads((self.run_dir/"summary.json").read_text())
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(all(summary["checks"].values()))
        self.assertEqual(summary["rows"], 335)

    def test_recommendation_command_respects_history(self):
        result = subprocess.run([sys.executable, "-m", "forum", "recommend", "--checkpoint",
            str(self.run_dir/"serving.pt"), "--user", "0", "--history", "[[12,4.0],[8,3.0]]", "--count", "5"],
            cwd=ROOT, capture_output=True, text=True, check=True)
        actions = json.loads(result.stdout)
        self.assertEqual(len(set(actions)), 5)
        self.assertFalse(set(actions) & {12, 8})

    def test_toy_generator_does_not_overwrite(self):
        before = self.data.read_bytes()
        result = subprocess.run([sys.executable, str(ROOT/"scripts/make_toy_data.py"), "--out", str(self.data)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.data.read_bytes(), before)
