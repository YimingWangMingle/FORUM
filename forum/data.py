"""Causal histories, censor-aware windows, action matching and session bootstrap.

Only training inputs are accepted here. Evaluation matrices have a separate CLI.
Feedback is a scalar and reward rho(y)=y in this version.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch

from .utils import sha256


@dataclass
class Session:
    session_id: str
    user: int
    actions: list
    feedback: list
    terminated: bool


def read_jsonl(path):
    sessions = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row["terminated"], bool):
                raise ValueError("terminated must be an explicit boolean")
            sessions.append(Session(**row))
    return sessions, {"source": str(Path(path).resolve()), "sha256": sha256(path),
                      "order": "caller-provided chronological sessions", "split": "training input only"}


def coat_sessions(train_path, order_seed=20261024, max_users=None):
    """Accept one training matrix; no discovery or reading of sibling files."""
    ratings = np.loadtxt(train_path, dtype=np.int64)
    if ratings.shape != (290, 300) or not np.isin(ratings, np.arange(6)).all():
        raise ValueError("Expected original Coat 290x300 training ratings")
    users = range(290 if max_users is None else min(max_users, 290))
    sessions = []
    for u in users:
        a = np.flatnonzero(ratings[u] > 0)
        a = np.random.default_rng(np.random.SeedSequence([order_seed, u])).permutation(a)
        sessions.append(Session(f"coat-{u}", u, a.tolist(), ratings[u, a].astype(float).tolist(), False))
    return sessions, dict(dataset="Coat", sha256=sha256(train_path), source=str(Path(train_path).resolve()),
        order="fixed artificial per-user permutation; NO observed timestamps", order_seed=order_seed,
        termination="rating exhaustion is right censoring, not user departure", training_only=True,
        genuine_sequential_benchmark=False, split="archived train.ascii; no new validation split")


def synthetic_sessions(seed=7, users=16, items=40, length=12):
    rng = np.random.default_rng(seed)
    sessions = []
    for u in range(users):
        for episode in range(2):
            # Shared first action ensures nontrivial same-action pairs.
            tail = rng.choice(np.arange(1, items), length - 1, replace=False)
            actions = [0, *tail.tolist()]
            mood = 1 if u % 2 else -1
            rewards = [float(np.clip(3 + mood + ((a + u) % 3 - 1), 1, 5)) for a in actions]
            ended = (u + episode) % 3 != 0
            if ended:
                n = int(rng.integers(6, length + 1)); actions, rewards = actions[:n], rewards[:n]
            sessions.append(Session(f"synthetic-{u}-{episode}", u, actions, rewards, ended))
    return sessions, dict(dataset="synthetic", seed=seed, known_terminal_and_censored_sessions=True)


class HistoryTable:
    def __init__(self, sessions, window, n_users=None, n_items=None, no_repeat=True):
        if not sessions or len({s.session_id for s in sessions}) != len(sessions):
            raise ValueError("Empty input or duplicate session identifiers")
        for s in sessions:
            if not isinstance(s.terminated, bool):
                raise ValueError("terminated must be an explicit boolean")
            if not s.actions or len(s.actions) != len(s.feedback):
                raise ValueError("Each session needs equal-length nonempty actions and feedback")
            if not isinstance(s.user, int) or s.user < 0 or any(not isinstance(a, int) or a < 0 for a in s.actions):
                raise ValueError("User and item IDs must be nonnegative integers")
            if not np.isfinite(s.feedback).all():
                raise ValueError("Non-finite feedback")
            if no_repeat and len(set(s.actions)) != len(s.actions):
                raise ValueError("NX_0 training input contains repeated actions; select a compatible protocol")
        self.sessions, self.window, self.no_repeat = sessions, window, no_repeat
        self.n_users = max(s.user for s in sessions) + 1 if n_users is None else n_users
        self.n_items = max(max(s.actions) for s in sessions) + 1 if n_items is None else n_items
        if max(s.user for s in sessions) >= self.n_users or max(max(s.actions) for s in sessions) >= self.n_items:
            raise ValueError("IDs exceed vocabulary")
        users, actions, feedback, lengths, seen = [], [], [], [], []
        state, next_state, reward, item, done, sid, turn = [], [], [], [], [], [], []
        self.offsets = []
        for si, s in enumerate(sessions):
            offset = len(users); self.offsets.append(offset)
            for t in range(len(s.actions) + 1):
                a = np.zeros(window, np.int64); y = np.zeros(window, np.float32)
                start = max(0, t - window); size = t - start
                a[:size] = np.asarray(s.actions[start:t]) + 1  # 0 is padding.
                y[:size] = s.feedback[start:t]
                mask = np.zeros(self.n_items, bool); mask[s.actions[:t]] = True
                users.append(s.user); actions.append(a); feedback.append(y); lengths.append(size); seen.append(mask)
                if t < len(s.actions):
                    state.append(offset + t); next_state.append(offset + t + 1)
                    reward.append(s.feedback[t]); item.append(s.actions[t])
                    done.append(s.terminated and t == len(s.actions)-1); sid.append(si); turn.append(t)
        self.users = torch.tensor(users)
        self.actions = torch.tensor(np.asarray(actions))
        self.feedback = torch.tensor(np.asarray(feedback))
        self.lengths = torch.tensor(lengths)
        self.seen = torch.tensor(np.asarray(seen))
        self.state = np.asarray(state); self.next_state = np.asarray(next_state)
        self.reward = np.asarray(reward, np.float32); self.item = np.asarray(item)
        self.done = np.asarray(done); self.session_index = np.asarray(sid); self.turn = np.asarray(turn)
        # A logging cutoff supplies feedback but not a negative departure label.
        self.terminal_observed = np.asarray([
            s.terminated or t < len(s.actions)-1
            for s in sessions for t in range(len(s.actions))], dtype=bool)

    def __len__(self):
        return len(self.item)

    def batch(self, ids, device):
        ids = torch.as_tensor(ids, dtype=torch.long)
        return tuple(v[ids].to(device) for v in (self.users, self.actions, self.feedback, self.lengths))

    def window_target(self, row, k, gamma):
        si, t = int(self.session_index[row]), int(self.turn[row]); s = self.sessions[si]
        available = len(s.actions) - t
        if available < k and not s.terminated:
            return None  # Never fabricate a censored future.
        n = min(k, available)
        total = sum(gamma**j * s.feedback[t+j] for j in range(n))
        # At a censored boundary S_{t,k} is unknown even though G and L are observed.
        survival = available > k
        successor = self.offsets[si] + t + n
        return total, n, survival, successor

    def pairs(self, horizons, gamma):
        by_action = {}
        for row, a in enumerate(self.item):
            targets = {}
            for k in sorted(set(horizons)):
                target = self.window_target(row, k, gamma)
                if target is not None:
                    targets[k] = target
            if targets:
                by_action.setdefault(int(a), []).append((row, targets))
        groups = [rows for rows in by_action.values() if len(rows) > 1]
        if not groups:
            raise ValueError("No admissible same-action pairs for representation training")
        return PairSampler(groups)

    def bootstrap_rows(self, rng):
        chosen = rng.integers(len(self.sessions), size=len(self.sessions))
        rows = np.concatenate([np.flatnonzero(self.session_index == s) for s in chosen])
        return rows, chosen.tolist()

    def logged_returns(self, gamma):
        """Observed suffix + bootstrap at a censored boundary; terminal tail is zero."""
        returns, discounts, ends = [], [], []
        for si, t in zip(self.session_index, self.turn):
            s = self.sessions[si]; n = len(s.actions)-int(t)
            returns.append(sum(gamma**j*s.feedback[t+j] for j in range(n)))
            exhausted = self.no_repeat and len(set(s.actions)) == self.n_items
            discounts.append(0. if s.terminated or exhausted else gamma**n)
            ends.append(self.offsets[si] + len(s.actions))
        return np.asarray(returns, np.float32), np.asarray(discounts, np.float32), np.asarray(ends)


class PairSampler:
    def __init__(self, groups):
        self.groups = groups
        # Admissible horizons are nested; all rows share the smallest horizon.
        # Select pairs uniformly FIRST, then a shared horizon, as in Eq. (8).
        weights = np.asarray([len(rows)*(len(rows)-1) for rows in groups], np.float64)
        self.probs = weights / weights.sum()

    def sample(self, rng, batch):
        result = {k: [] for k in ("left", "right", "horizon", "g_left", "g_right", "l_left", "l_right", "mask", "next_left", "next_right")}
        for g in rng.choice(len(self.groups), batch, p=self.probs):
            rows = self.groups[g]; i, j = rng.choice(len(rows), 2, replace=False)
            left, left_targets = rows[i]; right, right_targets = rows[j]
            k = int(rng.choice(sorted(left_targets.keys() & right_targets.keys())))
            a, b = left_targets[k], right_targets[k]
            values = (left, right, k, a[0], b[0], a[1], b[1], a[2] and b[2], a[3], b[3])
            for key, value in zip(result, values):
                result[key].append(value)
        return {k: np.asarray(v) for k, v in result.items()}
