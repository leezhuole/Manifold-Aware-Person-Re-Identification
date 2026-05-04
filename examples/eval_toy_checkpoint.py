# encoding: utf-8
"""Post-training toy eval: load ``train_toy_lmono.py`` checkpoint, bidirectional mAP, Spearman, D8."""

from __future__ import absolute_import, print_function

import argparse
import os.path as osp
import sys
import time
from datetime import datetime

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

from bau import datasets
from bau import models
from bau.evaluators import bidirectional_evaluate, spearman_rho_theta_severity
from bau.utils.data.preprocessor import Preprocessor
from bau.utils.data import transforms as T
from bau.utils.osutils import mkdir_if_missing
from bau.utils.serialization import load_checkpoint
from bau.utils.stdout_tee import StdoutTee


def _parse_alpha_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _resolve_log_path(args):
    if args.log_file and str(args.log_file).strip():
        p = osp.abspath(osp.expanduser(args.log_file.strip()))
        d = osp.dirname(p)
        if d:
            mkdir_if_missing(d)
        return p
    log_dir = osp.abspath(osp.expanduser(args.log_dir))
    mkdir_if_missing(log_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = osp.splitext(osp.basename(args.resume))[0]
    return osp.join(log_dir, "eval_toy_checkpoint_{}_{}.log".format(base, stamp))


def build_eval_loader(eval_rows, batch_size, workers, height, width):
    normalizer = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    test_t = T.Compose(
        [
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer,
        ]
    )
    eval_list = sorted(eval_rows)
    return DataLoader(
        Preprocessor(eval_list, root=None, transform=test_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )


def run_eval(model, dataset, batch_size, workers, height, width, alpha_values, print_freq):
    query_clean = sorted(dataset.query_s1)
    gallery_corrupted = sorted(dataset.gallery)
    eval_rows = sorted(set(query_clean) | set(gallery_corrupted))
    loader = build_eval_loader(eval_rows, batch_size, workers, height, width)
    metrics = bidirectional_evaluate(
        model,
        loader,
        query_clean=query_clean,
        gallery_corrupted=gallery_corrupted,
        query_corrupted=gallery_corrupted,
        gallery_clean=query_clean,
        alpha_values=alpha_values,
        return_theta=True,
        print_freq=print_freq,
        verbose=True,
    )
    theta_dict = metrics.pop("theta_by_fname", {})
    paths_for_rho = [t[0] for t in eval_rows]
    rho = spearman_rho_theta_severity(theta_dict, paths_for_rho)
    return metrics, rho


def _fmt_d8(d8):
    if not d8:
        return "D8=<empty>"
    mg = d8.get("mean_gap")
    mg_txt = "nan" if mg is None or (isinstance(mg, float) and mg != mg) else "{:.6f}".format(mg)
    parts = ["mean_gap={}".format(mg_txt), "n_pairs={}".format(d8.get("n_pairs", 0))]
    by_sev = d8.get("mean_gap_by_severity") or {}
    if by_sev:
        sev_bits = []
        for k in sorted(by_sev.keys()):
            v = by_sev[k]
            if v is None or (isinstance(v, float) and v != v):
                sev_bits.append("{}:nan".format(k))
            else:
                sev_bits.append("{}:{:.6f}".format(k, v))
        parts.append("by_severity={" + ", ".join(sev_bits) + "}")
    return "D8 " + " ".join(parts)


def main():
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is required.", file=sys.stderr)
        sys.exit(1)

    log_path = _resolve_log_path(args)
    print("Logging to: {}".format(log_path), file=sys.stderr)
    log_fp = open(log_path, "a", buffering=1)
    old_stdout = sys.stdout
    sys.stdout = StdoutTee(old_stdout, log_fp)
    try:
        print("========== eval_toy_checkpoint ==========")
        print(
            "time_utc={} log_file={}".format(
                time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), log_path
            )
        )
        print("argv={}".format(sys.argv))
        print("args={}".format(vars(args)))

        cudnn.benchmark = True

        dataset = datasets.create("toy_corruption", args.data_dir)
        model = models.create("toy_resnet50", num_classes=0, pretrained=False)
        model.cuda()

        ckpt_obj = load_checkpoint(args.resume)
        if not isinstance(ckpt_obj, dict) or "model" not in ckpt_obj:
            print("ERROR: --resume must be a dict with key 'model' (train_toy_lmono save format).", file=sys.stderr)
            sys.exit(1)
        missing, unexpected = model.load_state_dict(ckpt_obj["model"], strict=False)
        if missing:
            print("load_state_dict missing keys: {}".format(missing))
        if unexpected:
            print("load_state_dict unexpected keys: {}".format(unexpected))

        for par in model.parameters():
            par.requires_grad_(False)
        model.eval()

        alpha_values = _parse_alpha_list(args.alpha)
        metrics, rho = run_eval(
            model,
            dataset,
            args.eval_batch_size,
            args.workers,
            args.height,
            args.width,
            alpha_values,
            args.eval_print_freq,
        )

        print("Spearman rho(theta, severity) = {:.4f}".format(rho))
        print(
            "(L_mono target: strong negative correlation with severity index 0=clean..4=worst, e.g. rho <= -0.8)"
        )
        eu = metrics["euclidean"]
        print(
            "Euclidean mAP A={:.4f} B={:.4f} Δ={:.4f}".format(
                eu["direction_A"]["mAP"],
                eu["direction_B"]["mAP"],
                eu["delta_mAP"],
            )
        )
        for a in alpha_values:
            ra = metrics["randers"][a]
            d8 = ra.get("d8_mean_asymmetric_gap", {})
            print(
                "Randers a={} mAP A={:.4f} B={:.4f} Δ={:.4f} | {}".format(
                    a,
                    ra["direction_A"]["mAP"],
                    ra["direction_B"]["mAP"],
                    ra["delta_mAP"],
                    _fmt_d8(d8),
                )
            )
        print("========== eval_toy_checkpoint finished ==========")
    finally:
        sys.stdout = old_stdout
        log_fp.close()
        print("Log closed: {}".format(log_path), file=old_stdout)


parser = argparse.ArgumentParser(
    description="ToyCorruption eval from train_toy_lmono checkpoint (mAP, Spearman, D8)"
)
parser.add_argument("--resume", type=str, required=True, help="Path to .pth from train_toy_lmono.py")
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--eval-batch-size", type=int, default=64)
parser.add_argument(
    "--workers",
    type=int,
    default=0,
    help="DataLoader workers (0 avoids fork+CUDA issues).",
)
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--width", type=int, default=128)
parser.add_argument(
    "--alpha",
    type=str,
    default="0.0,0.1,0.3,0.5,0.9",
    help="Comma-separated Randers α values.",
)
parser.add_argument("--eval-print-freq", type=int, default=100)
parser.add_argument(
    "--log-dir",
    type=str,
    default="logs/toy_lmono",
    help="Directory for auto-named log files when --log-file is not set.",
)
parser.add_argument(
    "--log-file",
    type=str,
    default="",
    help="Explicit log path (stdout tee'd here for the whole run).",
)

if __name__ == "__main__":
    main()
