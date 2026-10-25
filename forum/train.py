"""Stage-ordered orchestration with frozen components and provenance artifacts."""
from pathlib import Path
import time
import platform
import shutil
import numpy as np
import torch

from .data import HistoryTable
from .policy import fit_policy
from .representation import train_representation
from .utils import log, seed_all, sha256, write_json
from .world import train_ensemble


def run_training(sessions, provenance, config, output, device="cpu", n_users=None, n_items=None,
                 feedback_values=None):
    config.validate(); start = time.monotonic(); out = Path(output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {out}")
    table = HistoryTable(sessions, config.history_window, n_users, n_items, config.no_repeat)
    # Validate supervision before creating a run directory.
    table.pairs(config.horizons, config.gamma)
    out.mkdir(parents=True)
    write_json(out/"config.json", config.to_dict()); write_json(out/"input.json", provenance)
    source_root = Path(__file__).resolve().parent
    hashes = {str(p.relative_to(source_root.parent)): sha256(p) for p in sorted(source_root.glob("*.py"))}
    write_json(out/"source_hashes.json", hashes)
    shutil.copytree(source_root, out/"source"/"forum", ignore=shutil.ignore_patterns("__pycache__"))
    seed_all(config.seed, config.threads)
    timings = {}; t = time.monotonic()
    encoder, latents, rep_history = train_representation(table, config, device)
    timings["representation_seconds"] = time.monotonic()-t
    frozen_encoder = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    t = time.monotonic()
    models, world_history, bootstraps = train_ensemble(table, latents, encoder, config, device, feedback_values)
    timings["world_seconds"] = time.monotonic()-t
    frozen_worlds = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in models]
    t = time.monotonic()
    policy = fit_policy(table, latents, encoder, models, config, device)
    timings["policy_seconds"] = time.monotonic()-t
    saved = dict(config=config.to_dict(), n_users=table.n_users, n_items=table.n_items,
                 encoder=encoder.state_dict(), behavior=policy["behavior"].state_dict(), actor=policy["actor"].state_dict())
    torch.save(saved, out/"serving.pt")
    torch.save(dict(**saved, worlds=[m.state_dict() for m in models], feedback_values=feedback_values,
                    raw=policy["raw"].state_dict(), penalized=policy["penalized"].state_dict(),
                    target_raw=policy["target_raw"].state_dict(), target_penalized=policy["target_penalized"].state_dict()),
               out/"training.pt")
    write_json(out/"bootstrap_sessions.json", [[table.sessions[i].session_id for i in row] for row in bootstraps])
    write_json(out/"training_history.json", dict(representation=rep_history, world=world_history, policy=policy["history"]))
    batch = policy["last_batch"]
    np.savez_compressed(out/"last_rollouts.npz", **{k: v.cpu().numpy() for k, v in batch.items()})
    checks = dict(
        encoder_frozen=all(torch.equal(frozen_encoder[k], v) for k, v in encoder.state_dict().items()),
        worlds_frozen=all(torch.equal(old[k], v) for m, old in zip(models, frozen_worlds) for k, v in m.state_dict().items()),
        shared_frozen_feedback_embedding=all(torch.equal(v, encoder.feedback_embedding.state_dict()[k])
            for m in models for k, v in m.feedback_embedding.state_dict().items()),
        raw_and_penalized_critics_separate=all(a.data_ptr() != b.data_ptr() for a, b in zip(policy["raw"].parameters(), policy["penalized"].parameters())),
        finite_model_parameters=all(torch.isfinite(v).all().item() for m in [encoder, *models, policy["actor"], policy["raw"], policy["penalized"]] for v in m.parameters()),
        nonnegative_penalty=bool((batch["penalty"] >= 0).all()),
        finite_rollouts=all(torch.isfinite(v).all().item() for v in batch.values()),
        trajectory_member_fixed=all(batch["member"][batch["trajectory"] == i].unique().numel() == 1 for i in batch["trajectory"].unique()),
        nx0_respected=not bool(batch["seen"].gather(1, batch["action"][:, None]).any()),
        terminal_successor_absorbing=bool((batch["next_z"][batch["done"]] == 0).all()),
        terminal_causes_recorded=bool(torch.equal(batch["done"], batch["model_done"] | batch["catalog_exhausted"])),
        exhausted_continuation_zero=bool((batch["u_value"][batch["catalog_exhausted"]] == 0).all()),
        sources_unchanged=all(sha256(source_root.parent/k) == v for k, v in hashes.items()),
    )
    from .serving import Recommender
    serving = Recommender(out/"serving.pt", device)
    with torch.no_grad():
        ids = np.arange(min(16, len(table.users)))
        loaded_z = serving.encoder(*table.batch(ids, device))
        checks["reload_encoding_exact"] = torch.equal(loaded_z, encoder(*table.batch(ids, device)))
        checks["reload_actor_exact"] = torch.equal(serving.actor(loaded_z), policy["actor"](loaded_z))
    users = sorted({s.user for s in sessions})[:8]
    histories = [[] for _ in users]
    for _ in range(min(5, table.n_items)):
        choices = serving.rank(users, histories)
        # Fixed observed feedback placeholder only exercises the serving interface, not a benchmark.
        for history, chosen in zip(histories, choices):
            history.append((chosen[0], 3.))
    checks["serving_nx0"] = all(len({a for a, _ in h}) == len(h) for h in histories) if config.no_repeat else True
    checks["serving_without_world_or_critic"] = not any(hasattr(serving, name) for name in ("worlds", "raw", "penalized"))
    summary = dict(status="passed" if all(checks.values()) else "failed", scope="training/inference validation; not a paper performance reproduction",
        rows=len(table), sessions=len(sessions), users=len({s.user for s in sessions}), catalog=table.n_items,
        true_terminals=int(table.done.sum()), censored_sessions=sum(not s.terminated for s in sessions),
        observed_termination_labels=int(table.terminal_observed.sum()),
        observed_nonterminal_transitions=int((table.terminal_observed & ~table.done).sum()),
        ensemble_size=len(models), simulated_transitions=policy["simulated"], simulated_terminals=policy["terminals"],
        checks=checks, timings=timings, wall_seconds=time.monotonic()-start,
        environment=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, device=str(device)),
        checkpoints={name: sha256(out/name) for name in ("serving.pt", "training.pt")},
        evaluation_data_used_for_fitting=False, validation_or_test_used_for_selection=False)
    write_json(out/"summary.json", summary)
    log("complete", **summary)
    if not all(checks.values()):
        raise AssertionError("Training invariant check failed; inspect summary.json")
    return summary
