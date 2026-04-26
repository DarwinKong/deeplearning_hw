#!/usr/bin/env python3
"""
读取多个 SourceTorch 实验目录下的 logs/*.csv，在终端打印对比（无 pandas 依赖）。

示例:
  python tools/compare_sourcetorch_experiments.py \\
    "checkpoints-and-logs/local/A2C 2026-04-18 17-46-37" \\
    "checkpoints-and-logs/local/PPO 2026-04-18 17-42-20"
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Optional


def _read_last_row(path: Path) -> Optional[Dict[str, str]]:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return rows[-1]


def summarize_exp(exp_dir: Path) -> Dict[str, str]:
    out: dict[str, str] = {"experiment": exp_dir.name}
    logs = exp_dir / "logs"
    hist = logs / "training_history_full.csv"
    ev = logs / "evaluation_metrics.csv"
    last = _read_last_row(hist)
    if last:
        for k, v in last.items():
            if any(x in k.lower() for x in ("loss", "entropy", "lr")):
                out[f"last_{k}"] = v
    last_e = _read_last_row(ev)
    if last_e:
        for k, v in last_e.items():
            out[f"eval_{k}"] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="对比 SourceTorch 实验 CSV 日志")
    parser.add_argument("experiment_dirs", nargs="+", help="实验根目录（含 logs/）")
    args = parser.parse_args()
    all_keys: list[str] = []
    rows: list[dict[str, str]] = []
    for d in args.experiment_dirs:
        p = Path(d)
        if not p.is_dir():
            print(f"跳过（非目录）: {d}", file=sys.stderr)
            continue
        r = summarize_exp(p)
        rows.append(r)
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    if not rows:
        print("无有效实验目录", file=sys.stderr)
        sys.exit(1)
    # 简单表格
    header = all_keys
    print("\t".join(header))
    for r in rows:
        print("\t".join(r.get(k, "") for k in header))


if __name__ == "__main__":
    main()
