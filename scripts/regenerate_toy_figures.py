"""
Regenerate camera-ready toy analysis figures from pre-computed JSON results.

Produces two files in paper/draft_4/fig (or --output-dir):
  fig_corruption_strip.pdf  -- image severity axis, no suptitle/caption
  fig_retrieval_mAP.pdf     -- mAP vs. severity, Finsler dF vs. Euclidean dE

Does NOT require GPU: reads retrieval_metrics.json and image files from disk.

Usage (from repo root):
    python scripts/regenerate_toy_figures.py
        [--results-dir-1a1c results/toy_analysis_1a1c]
        [--dataset-dir     examples/data/ToyCorruption]
        [--output-dir      paper/draft_4/fig]
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

# Allow `python scripts/regenerate_toy_figures.py` from repo root
_SCRIPTS_DIR = osp.dirname(osp.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from toy_paper_figures import copy_paper_outputs, make_corruption_strip, make_retrieval_mAP_plot


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default="results/toy_analysis",
        help="Directory containing retrieval_metrics.json (primary toy analysis output)",
    )
    parser.add_argument(
        "--dataset-dir",
        default="examples/data/ToyCorruption",
        help="ToyCorruption dataset directory (must contain bounding_box_test/)",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/draft_4/fig",
        help="Destination directory for the two output PDFs",
    )
    parser.add_argument(
        "--also-mirror-to",
        default="",
        help="If set, copy every file from --results-dir into this directory (paper bundle).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    retrieval_json = osp.join(args.results_dir, "retrieval_metrics.json")
    strip_out = osp.join(args.output_dir, "fig_corruption_strip.pdf")
    map_out = osp.join(args.output_dir, "fig_retrieval_mAP.pdf")

    make_corruption_strip(dataset_dir=args.dataset_dir, output_path=strip_out)
    print(f"Saved corruption strip → {strip_out}")

    make_retrieval_mAP_plot(retrieval_json, None, map_out)
    print(f"Saved retrieval mAP figure → {map_out}")

    if args.also_mirror_to.strip():
        copy_paper_outputs(args.results_dir, args.output_dir, mirror_dir=args.also_mirror_to.strip())


if __name__ == "__main__":
    main()
