# encoding: utf-8
"""PLAN.md **M1**: frozen Euclidean backbone, no θ head, no training — bidirectional toy eval only."""

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
from bau.evaluators import bidirectional_evaluate
from bau.utils.data.preprocessor import Preprocessor
from bau.utils.data import transforms as T
from bau.utils.osutils import mkdir_if_missing
from bau.utils.serialization import copy_state_dict, load_checkpoint
from bau.utils.stdout_tee import StdoutTee


def _state_dict_from_checkpoint(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj:
            return ckpt_obj["state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
    return ckpt_obj


def _strip_module_prefix(sd):
    if not isinstance(sd, dict):
        return sd
    if not any(k.startswith("module.") for k in sd):
        return sd
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def _resolve_log_path(p):
    if p.log_file and str(p.log_file).strip():
        path = osp.abspath(osp.expanduser(p.log_file.strip()))
        d = osp.dirname(path)
        if d:
            mkdir_if_missing(d)
        return path
    log_dir = osp.abspath(osp.expanduser(p.log_dir))
    mkdir_if_missing(log_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return osp.join(log_dir, "eval_toy_m1_{}.log".format(stamp))


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
    return DataLoader(
        Preprocessor(sorted(eval_rows), root=None, transform=test_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )


def main():
    p = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is required.", file=sys.stderr)
        sys.exit(1)

    log_path = _resolve_log_path(p)
    print("Logging to: {}".format(log_path), file=sys.stderr)
    log_fp = open(log_path, "a", buffering=1)
    old_stdout = sys.stdout
    sys.stdout = StdoutTee(old_stdout, log_fp)
    try:
        print("========== eval_toy_m1 ==========")
        print("time_utc={} log_file={}".format(
            time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), log_path
        ))
        print("argv={}".format(sys.argv))
        print("args={}".format(vars(p)))

        cudnn.benchmark = True

        dataset = datasets.create("toy_corruption", p.data_dir)
        model = models.create(
            "resnet50", num_classes=0, pretrained=False, with_theta_head=False
        )
        model.cuda()
        ckpt = load_checkpoint(p.checkpoint)
        sd = _strip_module_prefix(_state_dict_from_checkpoint(ckpt))
        copy_state_dict(sd, model, strip=None)
        for par in model.parameters():
            par.requires_grad_(False)
        model.eval()

        query_clean = sorted(dataset.query_s1)
        gallery_corrupted = sorted(dataset.gallery)
        eval_rows = sorted(set(query_clean) | set(gallery_corrupted))
        loader = build_eval_loader(
            eval_rows, p.eval_batch_size, p.workers, p.height, p.width
        )
        alpha_values = [float(x.strip()) for x in p.alpha.split(",") if x.strip()]
        metrics = bidirectional_evaluate(
            model,
            loader,
            query_clean=query_clean,
            gallery_corrupted=gallery_corrupted,
            query_corrupted=gallery_corrupted,
            gallery_clean=query_clean,
            alpha_values=alpha_values,
            return_theta=False,
            print_freq=p.eval_print_freq,
            verbose=True,
        )
        eu = metrics["euclidean"]
        print(
            "M1 Euclidean: mAP A={:.4f} B={:.4f} Δ={:.4f}".format(
                eu["direction_A"]["mAP"],
                eu["direction_B"]["mAP"],
                eu["delta_mAP"],
            )
        )
        for a in alpha_values:
            ra = metrics["randers"][a]
            print(
                "M1 Randers a={} (θ=0): mAP A={:.4f} B={:.4f} Δ={:.4f}".format(
                    a,
                    ra["direction_A"]["mAP"],
                    ra["direction_B"]["mAP"],
                    ra["delta_mAP"],
                )
            )
        print("========== eval_toy_m1 finished ==========")
    finally:
        sys.stdout = old_stdout
        log_fp.close()
        print("Log closed: {}".format(log_path), file=old_stdout)


parser = argparse.ArgumentParser(description="ToyCorruption M1 zero-shot Euclidean eval")
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
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
    help="Comma-separated α values (ranking matches Euclidean for all α when θ=0).",
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
