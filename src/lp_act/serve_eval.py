"""Serve standard ACT to the official BEHAVIOR 2026 websocket evaluator."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .eval_policy import ACTEvalPolicy
from .r1pro import ACTION_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("standard", "lp"), default="standard")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute-prefix", type=int, default=6)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--diagnostic-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from omnigibson.eval.utils.network_utils import WebsocketPolicyServer

    policy = ACTEvalPolicy(
        args.checkpoint,
        device=args.device,
        execute_prefix=args.execute_prefix,
        diagnostic_dir=args.diagnostic_dir,
        mode=args.mode,
    )
    metadata = {
        "policy": "standard_act" if args.mode == "standard" else "lp_act_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": policy.checkpoint_step,
        "execute_prefix": args.execute_prefix,
        "action_dim": ACTION_DIM,
    }
    WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
