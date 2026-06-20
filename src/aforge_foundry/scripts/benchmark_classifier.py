#!/usr/bin/env python
"""Benchmark and evaluate the creature classifier.

Two modes:

* ``throughput`` — measures end-to-end images/sec for the embedding pre-pass
  (cold, GPU-bound) and for the per-file classify pass (warm, cache hits),
  plus peak VRAM.  Use it to size batch sizes and project the wall-clock for
  the full 6,000-asset run.

* ``eval`` — measures **accuracy** against a hand-labelled validation set and
  reports the confidence/​margin operating point: how many assets clear the
  auto-accept threshold, and what the accuracy is on the confident vs the
  uncertain (VLM-fallback) tail.  This is how you choose
  ``--confidence-threshold`` honestly rather than guessing.

Validation-set format (``eval`` mode)
-------------------------------------
A CSV with columns ``path,asset_type,category`` — 150-300 hand-labelled
images spanning all 14 types and both asset formats is enough to estimate the
operating point with useful precision.

Examples
--------
    python scripts/benchmark_classifier.py throughput -i assets --batch-size 64
    python scripts/benchmark_classifier.py throughput --clip-model clip
    python scripts/benchmark_classifier.py eval --labels val.csv --threshold 0.85
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

from aforge_flow.config import ClipModelChoice, Config
from aforge_flow.processors.creature_classification import (
    CreatureClassificationProcessor,
)
from aforge_flow.utils import get_files_recursively


def _make_processor(args) -> CreatureClassificationProcessor:
    config = Config(
        input_dir=Path(args.input),
        output_dir=Path(args.input) / "_bench_out",
        classification_enabled=True,
        clip_model=ClipModelChoice(args.clip_model),
        classification_batch_size=args.batch_size,
        classification_sidecar=False,
        use_gpu=not args.cpu,
    )
    return CreatureClassificationProcessor(config)


def _peak_vram_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        pass
    return 0.0


def run_throughput(args) -> None:
    proc = _make_processor(args)
    files = get_files_recursively(Path(args.input))[: args.limit] if args.limit else (
        get_files_recursively(Path(args.input))
    )
    files = [f for f in files if proc.can_process(f)]
    if not files:
        print("No images found.")
        return

    print(f"Encoder: {proc.encoder.hf_id} (dim={proc.encoder.dim})")
    print(f"Images:  {len(files)}  |  batch_size={args.batch_size}")

    t0 = time.perf_counter()
    computed = proc.precompute_embeddings(files)
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    for f in files:
        proc.process(f)
    t_classify = time.perf_counter() - t0

    embed_rate = (computed / t_embed) if t_embed else float("inf")
    classify_rate = len(files) / t_classify if t_classify else float("inf")

    print("\n=== Throughput ===")
    print(f"Embedding pre-pass : {computed} new in {t_embed:6.2f}s  -> {embed_rate:7.1f} img/s (GPU)")
    print(f"Classify pass      : {len(files)} in {t_classify:6.2f}s  -> {classify_rate:7.1f} img/s (cached)")
    print(f"Peak VRAM          : {_peak_vram_gb():.2f} GB")
    if embed_rate not in (0, float("inf")):
        eta = 6000 / embed_rate
        print(f"Projected 6,000-asset cold embed: ~{eta/60:.1f} min")


def _load_labels(csv_path: Path) -> List[Tuple[Path, str, str]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((Path(row["path"]), row["asset_type"].strip(), row["category"].strip()))
    return rows


def run_eval(args) -> None:
    proc = _make_processor(args)
    labels = _load_labels(Path(args.labels))
    proc.precompute_embeddings([p for p, _, _ in labels])

    threshold = args.threshold
    margin_threshold = args.margin

    stats: Dict[str, int] = dict(
        n=0, type_correct=0, cat_correct=0,
        confident=0, conf_cat_correct=0,
        uncertain=0, unc_cat_correct=0,
    )
    for path, gt_type, gt_cat in labels:
        if not path.exists():
            continue
        res = proc.process(path)
        m = res.metadata
        stats["n"] += 1
        stats["type_correct"] += int(m["asset_type"] == gt_type)
        cat_ok = int(m["category"] == gt_cat)
        stats["cat_correct"] += cat_ok
        confident = m["confidence"] >= threshold and m["category_margin"] >= margin_threshold
        if confident:
            stats["confident"] += 1
            stats["conf_cat_correct"] += cat_ok
        else:
            stats["uncertain"] += 1
            stats["unc_cat_correct"] += cat_ok

    n = max(stats["n"], 1)
    conf = max(stats["confident"], 1)
    unc = max(stats["uncertain"], 1)
    print("\n=== Evaluation ===")
    print(f"Samples              : {stats['n']}")
    print(f"Asset-type accuracy  : {stats['type_correct']/n:6.1%}")
    print(f"Category accuracy    : {stats['cat_correct']/n:6.1%}")
    print(f"\nOperating point (threshold={threshold}, margin={margin_threshold}):")
    print(f"  Auto-accepted      : {stats['confident']/n:6.1%} of assets")
    print(f"  Acc on accepted    : {stats['conf_cat_correct']/conf:6.1%}  <- the number users feel")
    print(f"  Sent to fallback   : {stats['uncertain']/n:6.1%} of assets")
    print(f"  Acc on uncertain   : {stats['unc_cat_correct']/unc:6.1%}  <- VLM headroom here")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-i", "--input", default="assets")
    common.add_argument("--clip-model", default="siglip2", choices=[m.value for m in ClipModelChoice])
    common.add_argument("--batch-size", type=int, default=32)
    common.add_argument("--cpu", action="store_true")

    p_tp = sub.add_parser("throughput", parents=[common])
    p_tp.add_argument("--limit", type=int, default=0, help="Cap number of images (0 = all)")

    p_ev = sub.add_parser("eval", parents=[common])
    p_ev.add_argument("--labels", required=True, help="CSV: path,asset_type,category")
    p_ev.add_argument("--threshold", type=float, default=0.85)
    p_ev.add_argument("--margin", type=float, default=0.10)

    args = parser.parse_args()
    if args.mode == "throughput":
        run_throughput(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
