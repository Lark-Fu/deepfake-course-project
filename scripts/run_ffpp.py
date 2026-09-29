#!/usr/bin/env python3
"""Entry point reserved for formal FaceForensics++ evaluation."""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FF++ after paths, checkpoints, and validation threshold are configured.")
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise SystemExit("FF++ runner is not enabled yet: first validate official checkpoint loading on one cropped face.")


if __name__ == "__main__":
    main()
