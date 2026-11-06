"""Expand verified bundled Knights panels and write a portable run configuration."""

import argparse
import json

from knights_recipe import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--model", required=True, help="Local HF checkpoint and tokenizer directory"
    )
    parser.add_argument(
        "--profile", choices=("production", "smoke"), default="production"
    )
    args = parser.parse_args()
    manifest = prepare(args.root, args.model, profile=args.profile)
    print(
        json.dumps(
            {
                "profile": manifest["profile"],
                "data": manifest["data"],
                "historical_initialization_verified": manifest[
                    "historical_initialization_verified"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
