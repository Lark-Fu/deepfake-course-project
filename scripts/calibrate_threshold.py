#!/usr/bin/env python3
"""Deprecated: formal UADFV evaluation uses the fixed 0.5 model threshold."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_utils import calibrate_f1


def main() -> int:
    raise SystemExit("Deprecated for this project: do not optimize a threshold on UADFV labels. Use 0.5.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True, help="Validation CSV with label and frame_probability columns.")
    parser.add_argument("--model", choices=("xception", "effort"), required=True)
    parser.add_argument("--output", type=Path, default=Path("configs/thresholds.yaml"))
    args = parser.parse_args()
    table = pd.read_csv(args.predictions)
    required = {"label", "frame_probability"}
    missing = required - set(table.columns)
    if missing:
        raise SystemExit(f"Missing CSV columns: {sorted(missing)}")
    threshold, f1 = calibrate_f1(table["label"].to_numpy(), table["frame_probability"].to_numpy())
    config = yaml.safe_load(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    config[args.model] = float(threshold)
    config["calibration_dataset"] = "FaceForensics++ validation"
    config["selection_rule"] = "maximum_f1"
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"{args.model} threshold={threshold:.6f}; validation_f1={f1:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
