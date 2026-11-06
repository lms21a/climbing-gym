"""Small, lazy dispatcher for the real native-engine workflows."""
from __future__ import annotations

import argparse
from importlib import import_module
import sys
from typing import Sequence

from . import __version__

_COMMANDS = {
    "train": ("train", "Train PPO, GRPO, CISPO, FlashREINFORCE, or MIS variants"),
    "evaluate": ("eval", "Evaluate a checkpoint and save complete model traces"),
    "sft": ("sft", "Supervised fine-tuning from text or exact token IDs"),
    "project": ("sampling", "Verified native greedy likelihood projection"),
    "preflight": ("preflight", "Prepare representative prompt IDs without model weights"),
    "benchmark": ("inference.benchmark", "Measure native throughput and peak memory"),
}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="climbing-gym",
        description="Native LFM post-training and verified reasoning environments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Workflows:\n" + "\n".join(
            f"  {name:10} {description}" for name, (_, description) in _COMMANDS.items()
        ) + "\n\nRun climbing-gym COMMAND --help for workflow options."),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("command", choices=_COMMANDS, metavar="COMMAND", nargs="?", help="workflow to run")
    if not arguments or arguments[0] in {"-h", "--help", "--version"}:
        parser.parse_args(arguments)
        if not arguments:
            parser.print_help()
        return 0
    command = parser.parse_args(arguments[:1]).command
    module = import_module(f"climbing_gym.{_COMMANDS[command][0]}")
    remaining = arguments[1:]
    return int(module.main(remaining) or 0)
