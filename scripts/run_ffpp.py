#!/usr/bin/env python3
"""Deprecated formal runner; FF++ Mini is allowed only for demo/debug."""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FF++ after paths, checkpoints, and validation threshold are configured.")
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise SystemExit("Deprecated: FF++ Mini must not be reported as formal generalization evaluation.")


if __name__ == "__main__":
    main()
