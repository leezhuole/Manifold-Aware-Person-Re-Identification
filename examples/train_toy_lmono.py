# encoding: utf-8
"""ToyCorruption L_mono / Randers training (PLAN.md Step 7).

Trains only ``theta_head`` + ``toy_classifier`` on frozen ResNet-50; evaluates
bidirectional mAP with ``bidirectional_evaluate`` and logs Spearman ρ(θ, severity).
"""

from __future__ import absolute_import, print_function

import argparse
import os
import os.path as osp
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from bau import datasets
from bau import models
from bau.evaluators import (
    bidirectional_evaluate,
    spearman_rho_theta_severity,
)
from bau.loss import CrossEntropyLabelSmooth, MonotonicityLoss, count_monotonicity_pairs
from bau.utils.data.preprocessor import Preprocessor, SeverityPreprocessor
from bau.utils.data.sampler import SeverityStratifiedSampler
from bau.utils.data import transforms as T
from bau.utils.osutils import mkdir_if_missing
from bau.utils.serialization import copy_state_dict, load_checkpoint
from bau.utils.stdout_tee import StdoutTee


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
    lm = str(args.lambda_mono).replace(".", "p")
    name = "train_toy_lmono_{}_seed{}_lmono{}.log".format(stamp, args.seed, lm)
    return osp.join(log_dir, name)


def _parse_alpha_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


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


def build_train_loader(
    train_set,
    batch_size,
    workers,
    height,
    width,
    num_instances,
    seed,
):
    normalizer = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_t = T.Compose(
        [
            T.Resize((height, width), interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
        ]
    )
    train_list = sorted(train_set)
    sampler = SeverityStratifiedSampler(train_list, batch_size, num_instances)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        SeverityPreprocessor(train_list, root=None, transform=train_t),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        generator=g,
    )


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


def run_eval(
    model,
    dataset,
    batch_size,
    workers,
    height,
    width,
    alpha_values,
    print_freq,
):
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


def train_one_epoch(
    model,
    toy_classifier,
    train_loader,
    criterion_ce,
    criterion_mono,
    optimizer,
    lambda_mono,
    epoch,
    log_interval,
):
    model.train()
    toy_classifier.train()
    ce_meter = 0.0
    mono_meter = 0.0
    n_batches = 0
    zero_pair_batches = 0
    theta_std_sum = 0.0

    for batch_idx, batch in enumerate(train_loader):
        imgs, _, pids, _, sevs = batch
        imgs = imgs.cuda()
        pids = pids.cuda()
        sevs = sevs.cuda()

        emb, f_norm, theta = model(imgs)
        logits = toy_classifier(f_norm)
        loss_ce = criterion_ce(logits, pids)
        theta_s = theta.view(-1)
        npairs = count_monotonicity_pairs(pids, sevs)
        if npairs == 0:
            zero_pair_batches += 1
        loss_mono = criterion_mono(theta_s, pids, sevs)
        loss = loss_ce + lambda_mono * loss_mono

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ce_meter += float(loss_ce.item())
        mono_meter += float(loss_mono.item())
        theta_std_sum += float(theta_s.detach().std().item())
        n_batches += 1

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            print(
                "Epoch {} [{}/{}] CE {:.4f} Mono {:.4f} theta_std {:.4f}".format(
                    epoch,
                    batch_idx + 1,
                    len(train_loader),
                    ce_meter / n_batches,
                    mono_meter / n_batches,
                    theta_std_sum / n_batches,
                )
            )

    return {
        "loss_ce": ce_meter / max(n_batches, 1),
        "loss_mono": mono_meter / max(n_batches, 1),
        "theta_std_mean": theta_std_sum / max(n_batches, 1),
        "zero_pair_frac": float(zero_pair_batches) / max(n_batches, 1),
    }


def main():
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required (CrossEntropyLabelSmooth uses GPU tensors).")
        sys.exit(1)

    log_path = _resolve_log_path(args)
    print("Logging to: {}".format(log_path), file=sys.stderr)
    log_fp = open(log_path, "a", buffering=1)
    old_stdout = sys.stdout
    sys.stdout = StdoutTee(old_stdout, log_fp)
    try:
        print("========== train_toy_lmono ==========")
        print("time_utc={} log_file={}".format(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), log_path))
        print("argv={}".format(sys.argv))
        print("args={}".format(vars(args)))
        if args.arch != "toy_resnet50":
            print(
                "Warning: this script is designed for --arch toy_resnet50 "
                "(got {}).".format(args.arch)
            )

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

        _main_train_loop(args)
    finally:
        sys.stdout = old_stdout
        log_fp.close()
        print("Log closed: {}".format(log_path), file=old_stdout)


def _main_train_loop(args):
    dataset = datasets.create("toy_corruption", args.data_dir)
    num_classes = int(dataset.num_train_pids)

    # False: BAU checkpoint fills weights; True can block on ImageNet download (no progress).
    model = models.create(args.arch, num_classes=0, pretrained=False)
    model.cuda()
    toy_classifier = nn.Linear(2048, num_classes, bias=False).cuda()

    ckpt = load_checkpoint(args.checkpoint)
    sd = _strip_module_prefix(_state_dict_from_checkpoint(ckpt))
    copy_state_dict(sd, model, strip=None)
    model.freeze_pretrained()

    criterion_ce = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1)
    criterion_mono = MonotonicityLoss(margin=args.margin)

    params = list(model.theta_head.parameters()) + list(toy_classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=5e-4)

    train_loader = build_train_loader(
        dataset.train,
        args.batch_size,
        args.workers,
        args.height,
        args.width,
        args.num_instances,
        args.seed,
    )

    alpha_values = _parse_alpha_list(args.alpha)

    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(
            model,
            toy_classifier,
            train_loader,
            criterion_ce,
            criterion_mono,
            optimizer,
            args.lambda_mono,
            epoch,
            args.log_interval,
        )
        print(
            "Epoch {} train: loss_ce={:.4f} loss_mono={:.4f} theta_std={:.4f} "
            "zero_pair_frac={:.4f}".format(
                epoch,
                stats["loss_ce"],
                stats["loss_mono"],
                stats["theta_std_mean"],
                stats["zero_pair_frac"],
            )
        )

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
        print("Epoch {} Spearman rho(theta, severity) = {:.4f}".format(epoch, rho))
        eu = metrics["euclidean"]
        print(
            "  Euclidean mAP A={:.4f} B={:.4f} Δ={:.4f}".format(
                eu["direction_A"]["mAP"],
                eu["direction_B"]["mAP"],
                eu["delta_mAP"],
            )
        )
        for a in alpha_values:
            ra = metrics["randers"][a]
            print(
                "  Randers a={} mAP A={:.4f} B={:.4f} Δ={:.4f}".format(
                    a,
                    ra["direction_A"]["mAP"],
                    ra["direction_B"]["mAP"],
                    ra["delta_mAP"],
                )
            )

    if args.save_path:
        os.makedirs(osp.dirname(args.save_path) or ".", exist_ok=True)
        torch.save(
            {
                "epoch": args.epochs,
                "model": model.state_dict(),
                "toy_classifier": toy_classifier.state_dict(),
                "args": vars(args),
            },
            args.save_path,
        )
        print("Wrote checkpoint to {}".format(args.save_path))

    print("========== train_toy_lmono finished ==========")


parser = argparse.ArgumentParser(description="ToyCorruption L_mono training")
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--lambda-mono", type=float, default=0.1)
parser.add_argument("--margin", type=float, default=0.1)
parser.add_argument(
    "--alpha",
    type=str,
    default="0.0,0.1,0.3,0.5,0.9",
    help="Comma-separated Randers α values for eval.",
)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--eval-batch-size", type=int, default=64)
parser.add_argument("--num-instances", type=int, default=5)
parser.add_argument("--lr", type=float, default=3.5e-4)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--workers",
    type=int,
    default=0,
    help="DataLoader workers (0 avoids fork+CUDA hangs on some systems).",
)
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--arch", type=str, default="toy_resnet50")
parser.add_argument("--log-interval", type=int, default=20)
parser.add_argument("--eval-print-freq", type=int, default=100)
parser.add_argument(
    "--save-path",
    type=str,
    default="",
    help="Optional path to save final model + toy_classifier state.",
)
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
    help="Explicit log path (all stdout, including eval prints, is tee'd here).",
)

if __name__ == "__main__":
    main()
