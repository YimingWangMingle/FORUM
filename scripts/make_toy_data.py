"""Generate synthetic JSONL sessions for exercising the ordinary training CLI."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forum.data import synthetic_sessions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    output = Path(args.out)
    sessions, _ = synthetic_sessions(seed=args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        for session in sessions:
            f.write(json.dumps(asdict(session), allow_nan=False) + "\n")
    print(f"Wrote {len(sessions)} synthetic sessions; n_users=16, n_items=40. Not benchmark data.")


if __name__ == "__main__":
    main()
