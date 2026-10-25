import argparse
import json

from .config import Config


def main():
    parser = argparse.ArgumentParser(description="FORUM standalone offline recommendation")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("smoke", "train-coat", "train"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True)
        sub.add_argument("--out", required=True)
        sub.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
        if command == "train-coat":
            sub.add_argument("--train-matrix", required=True)
            sub.add_argument("--max-users", type=int)
            sub.add_argument("--order-seed", type=int, default=20261024)
        elif command == "train":
            sub.add_argument("--sessions", required=True)
            sub.add_argument("--n-users", type=int, required=True)
            sub.add_argument("--n-items", type=int, required=True)
            sub.add_argument("--feedback-values", default=None, help="Comma-separated categories; omitted means Gaussian feedback")
    evaluation = commands.add_parser("evaluate-coat")
    evaluation.add_argument("--run", required=True); evaluation.add_argument("--matrix", required=True)
    evaluation.add_argument("--out", required=True); evaluation.add_argument("--device", default="cpu")
    recommend = commands.add_parser("recommend")
    recommend.add_argument("--checkpoint", required=True); recommend.add_argument("--user", type=int, required=True)
    recommend.add_argument("--history", default="[]", help="JSON [[item_id, observed_feedback], ...]")
    recommend.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    if args.command == "evaluate-coat":
        from .evaluate import evaluate_coat
        print(json.dumps(evaluate_coat(args.run, args.matrix, args.out, args.device), indent=2, ensure_ascii=False))
        return
    if args.command == "recommend":
        from .serving import Recommender
        print(json.dumps(Recommender(args.checkpoint).rank([args.user], [json.loads(args.history)], args.count)[0]))
        return
    from .data import coat_sessions, read_jsonl, synthetic_sessions
    from .train import run_training
    config = Config.load(args.config)
    if args.command == "smoke":
        sessions, provenance = synthetic_sessions()
        n_users, n_items, values = 16, 40, [1., 2., 3., 4., 5.]
    elif args.command == "train-coat":
        if args.max_users is not None and args.max_users < 2:
            parser.error("--max-users must be at least 2")
        sessions, provenance = coat_sessions(args.train_matrix, args.order_seed, args.max_users)
        n_users, n_items, values = 290, 300, [1., 2., 3., 4., 5.]
    else:
        sessions, provenance = read_jsonl(args.sessions)
        n_users, n_items = args.n_users, args.n_items
        values = None if args.feedback_values is None else [float(x) for x in args.feedback_values.split(",")]
        if values is not None and (not values or len(set(values)) != len(values)):
            parser.error("Feedback categories must be nonempty and unique")
    run_training(sessions, provenance, config, args.out, args.device, n_users, n_items, values)


if __name__ == "__main__":
    main()
