from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import random
import numpy as np
import sys
import gc
import collections
import geoopt
from typing import cast

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.backends import cudnn
from torch.utils.data import DataLoader

from bau import datasets
from bau import models
from bau.trainers import BAUTrainer
from bau.models.memory import MemoryBank
from bau.evaluators import Evaluator, extract_features
from bau.utils.data import IterLoader, MultiSourceTrainDataset
from bau.utils.data import transforms as T
from bau.utils.data.randaugment import RandAugment
from bau.utils.data.sampler import RandomIdentitySampler
from bau.utils.data.preprocessor import Preprocessor, TwoViewPreprocessor
from bau.utils.logging import Logger
from bau.utils.lr_scheduler import WarmupMultiStepLR

# Optional Weights & Biases integration (minimal, modular)
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

best_mAP = 0


def parse_optional_chunk_size(value):
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise argparse.ArgumentTypeError("manifold chunk size must be a positive integer or 'none'")
        return value
    value_str = value.strip()
    if value_str.lower() in {"none", "null"}:
        return None
    try:
        parsed = int(value_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("manifold chunk size must be a positive integer or 'none'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("manifold chunk size must be a positive integer or 'none'")
    return parsed


def get_data(name, data_dir):
    root = data_dir
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers, num_instances, images_dir):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    
    transforms_w = T.Compose([T.Resize((height, width), interpolation=3),
                              T.RandomHorizontalFlip(p=0.5),
                              T.Pad(10),
                              T.RandomCrop((height, width)),
                              T.ToTensor(),
                              normalizer,])

    transforms_s = T.Compose([T.Resize((height, width), interpolation=3),
                              T.RandomHorizontalFlip(p=0.5),
                              T.Pad(10),
                              T.RandomCrop((height, width)),
                              T.RandomApply([T.ColorJitter(brightness=(0.5, 2.0), contrast=(0.5, 2.0), saturation=(0.5, 2.0), hue=(-0.1, 0.1))], p=args.prob),
                              T.ToTensor(),
                              normalizer,
                              T.RandomErasing(mean=[0.485, 0.456, 0.406], probability=args.prob),])
    
    transforms_s.transforms.insert(0, T.RandomApply([RandAugment()], p=args.prob))

    train_set = sorted(dataset.train)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomIdentitySampler(train_set, batch_size, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(DataLoader(TwoViewPreprocessor(train_set, root=images_dir, transform_w=transforms_w, transform_s=transforms_s),
                                         batch_size=batch_size, num_workers=workers, sampler=sampler,shuffle=not rmgs_flag, pin_memory=True, 
                                         drop_last=True), length=None)
    return train_loader


def get_memory_loader(dataset, height, width, batch_size, workers, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])
    if trainset is None:
        trainset = list(set(dataset.query) | set(dataset.gallery))

    memory_loader = DataLoader(TwoViewPreprocessor(trainset, root=dataset.images_dir, transform=test_transformer),
                             batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=False)
    return memory_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])
    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))
    test_loader = DataLoader(Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
                             batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=False)
    return test_loader


def create_model(args, num_classes, manifold):
    model = models.create(args.arch, num_classes=num_classes, manifold=manifold)
    model.cuda()
    model = torch.nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global best_mAP
    cudnn.benchmark = True

    os.makedirs(args.logs_dir, exist_ok=True)
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    def log(message, level=1):
        if args.verbosity >= level:
            print(message, flush=True)

    # Initialize wandb if available
    if wandb is not None:
        wandb.init(project='BAU', config=vars(args), name=args.wandb_name or None)

    # Organize dataset
    source_dataset = []
    log("Loading source datasets...", level=1)
    for ds in args.source_dataset:
        dataset = get_data(ds, args.data_dir)
        pid_count = getattr(dataset, 'num_train_pids', 'unknown')
        log(f"Loaded source dataset '{ds}' with {pid_count} train IDs", level=2)
        source_dataset.append(dataset)
        
    train_dataset = MultiSourceTrainDataset(datasets=source_dataset)
    combined_pids = getattr(train_dataset, 'num_train_pids', 'unknown')
    log(f"Combined multi-source dataset with {combined_pids} train IDs", level=1)
    train_loader = get_train_loader(args, train_dataset, args.height, args.width, args.batch_size,
                                    args.workers, args.num_instances, train_dataset.images_dir)
    log("Built training data loader", level=2)
    
    memory_loader = get_memory_loader(train_dataset, args.height, args.width, args.batch_size, args.workers,
                                      trainset=sorted(train_dataset.train))
    log("Prepared memory loader for initial feature extraction", level=2)
    
    test_dataset = get_data(args.target_dataset, args.data_dir)
    target_pids = getattr(test_dataset, 'num_train_pids', 'n/a')
    log(f"Loaded target dataset '{args.target_dataset}' (train IDs: {target_pids})", level=1)
    test_loader = get_test_loader(test_dataset, args.height, args.width, args.batch_size, args.workers)
    log("Built test data loader", level=2)

    # Backbone
    if getattr(args, "manifold_aware", True):
        manifold = geoopt.PoincareBall(c=args.curvature)
        log(f"Using manifold-aware model with curvature {args.curvature}", level=1)
    else:
        manifold = None
        log("Using Euclidean model", level=1)
    num_classes = train_dataset.num_train_pids
    model = create_model(args, num_classes=num_classes, manifold=manifold)
    log("Created model and moved to device", level=2)

    if args.fine_tuning:
        if args.checkpoint_path and osp.exists(args.checkpoint_path):
            log(f"Loading checkpoint from {args.checkpoint_path} for fine-tuning", level=1)
            checkpoint = torch.load(args.checkpoint_path)
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            model_dict = model.state_dict()
            new_state_dict = {}
            
            for k, v in state_dict.items():
                target_k = None
                if k in model_dict:
                    target_k = k
                elif k.startswith('module.') and k[7:] in model_dict:
                    target_k = k[7:]
                elif 'module.' + k in model_dict:
                    target_k = 'module.' + k
                
                if target_k:
                    if model_dict[target_k].shape == v.shape:
                        new_state_dict[target_k] = v
                    else:
                        log(f"Skipping layer {k} due to shape mismatch: {v.shape} vs {model_dict[target_k].shape}", level=1)
            
            model.load_state_dict(new_state_dict, strict=False)
            
            # Freeze backbone
            if isinstance(model, torch.nn.DataParallel):
                mod = model.module
            else:
                mod = model
                
            if hasattr(mod, 'base'):
                for param in mod.base.parameters():
                    param.requires_grad = False
                log("Freezed backbone (base) layers", level=1)
            else:
                # Fallback: freeze everything except classifier and bn_neck
                log("Model has no 'base' attribute. Freezing everything except classifier and bn_neck.", level=1)
                for name, param in mod.named_parameters():
                    if 'classifier' not in name and 'bn_neck' not in name:
                        param.requires_grad = False
        else:
            log("Fine-tuning requested but checkpoint path is invalid or missing.", level=1)

    # Log dataset/model summary once
    if wandb is not None and getattr(wandb, "run", None) is not None:
        try:
            num_params = sum(p.numel() for p in model.parameters())
            wandb.log({
                'data/num_train_ids': train_dataset.num_train_pids,
                'data/num_train_imgs': train_dataset.num_train_imgs,
                'data/num_train_cams': train_dataset.num_train_cams,
                'data/num_sources': len(source_dataset),
                'model/num_parameters': num_params,
            })
        except Exception:
            pass

    # memory bank with init
    memory_bank = MemoryBank(2048, num_classes, manifold=manifold).cuda()
    bank_features = cast(torch.Tensor, memory_bank.features)
    bank_labels = cast(torch.Tensor, memory_bank.labels)
    device = bank_features.device

    log("Extracting initial features for memory bank", level=1)
    features, _ = extract_features(model, memory_loader, print_freq=50)
    log("Finished feature extraction", level=1)
    features_dict = collections.defaultdict(list)
    for f, pid, _, _ in sorted(train_dataset.train):
        features_dict[pid].append(features[f].unsqueeze(0))
    log("Computed feature lists per identity", level=2)
    centroids = []
    for pid in sorted(features_dict.keys()):
        reps = torch.cat(features_dict[pid], 0).to(device)
        if manifold is None:
            centroid = F.normalize(reps.mean(0), dim=0)
        else:
            reps = manifold.projx(reps, dim=-1)
            tangent = manifold.logmap0(reps, dim=-1)
            mean_tangent = tangent.mean(dim=0, keepdim=True)
            centroid = manifold.expmap0(mean_tangent, dim=-1)
            centroid = manifold.projx(centroid, dim=-1).squeeze(0)
        centroids.append(centroid.cpu())
    centroids = torch.stack(centroids, 0).to(device)
    memory_bank.features = centroids
    log("Initialized memory bank centroids", level=1)

    domain_offset = 0
    domain_labels = []
    for ds in source_dataset:
        domain_labels.append(torch.ones(ds.num_train_pids, dtype=torch.long)*domain_offset)
        domain_offset += 1
    domain_labels = torch.cat(domain_labels)
    memory_bank.labels = domain_labels.to(device=bank_labels.device)
    log("Assigned domain labels in memory bank", level=2)

    del source_dataset, memory_loader, features_dict, centroids, domain_labels
    gc.collect()
    log("Freed temporary dataset references", level=2)

    # evaluator
    evaluator = Evaluator(model)
    log("Initialized evaluator", level=2)

    # optimizer
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    # optimizer = torch.optim.Adam(params)
    optimizer = geoopt.optim.RiemannianAdam(
        params, 
        stabilize=10                            # Optional: helps numerical stability for hyperbolic params
    )

    lr_scheduler = WarmupMultiStepLR(optimizer, 
                                     args.milestones, 
                                     gamma=0.1, 
                                     warmup_factor=0.01, 
                                     warmup_iters=args.warmup_step)
    log("Optimizer and scheduler ready", level=2)

    # train
    trainer = BAUTrainer(model=model, 
                         memory_bank=memory_bank, 
                         num_classes=num_classes, 
                         margin=args.margin, 
                         lam=args.lam, 
                         k=args.k, 
                         manifold=manifold,
                         manifold_chunk_size=args.manifold_chunk_size,
                         use_aug_ce=args.use_aug_ce,
                         use_align=not args.no_align,
                         use_uniform=not args.no_uniform,
                         use_domain=not args.no_domain, 
                         use_triplet=not args.no_triplet,
                         use_ce=not args.no_ce)
    
    log("Trainer constructed", level=2)

    log(f"Starting training loop for {args.epochs} epochs", level=1)

    for epoch in range(args.epochs):
        log(f"Epoch {epoch + 1}/{args.epochs} - starting", level=2)
        train_loader.new_epoch() 
        trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq, iters=args.iters)
        log(f"Epoch {epoch + 1}/{args.epochs} - training step completed", level=2)
        lr_scheduler.step()
        log(f"Epoch {epoch + 1}/{args.epochs} - scheduler stepped", level=2)

        if (epoch+1 in args.eval_epochs) or (epoch == args.epochs-1):
            with torch.no_grad():
                print('\n* Finished epoch {:3d}'.format(epoch))
                log(f"Running evaluation after epoch {epoch + 1}", level=1)
                eval_outputs = evaluator.evaluate(test_loader, test_dataset.query, test_dataset.gallery, cmc_flag=False)
                if isinstance(eval_outputs, tuple):
                    mAP = float(eval_outputs[0])
                    visfig = eval_outputs[1]
                else:
                    mAP = float(eval_outputs)
                    visfig = None
                vis_path = None
                if visfig is not None:
                    fig = getattr(visfig, 'figure', None)
                    if fig is not None:
                        vis_path = osp.join(args.logs_dir, 'vis_epoch_{:03d}.png'.format(epoch))
                        fig.savefig(vis_path, bbox_inches='tight')
                        plt.close(fig)
                if wandb is not None and getattr(wandb, "run", None) is not None:
                    # log epoch-level aggregates
                    try:
                        log_dict = {'eval/mAP': mAP, 'epoch': epoch}
                        if vis_path is not None:
                            log_dict['eval/embedding_vis'] = wandb.Image(vis_path, caption='Latent Embeddings Epoch {:d}'.format(epoch))
                        wandb.log(log_dict)
                    except Exception:
                        pass
                if mAP > best_mAP:
                    best_mAP = mAP
                    torch.save(model.state_dict(), osp.join(args.logs_dir, 'best.pth'))
                    log(f"New best mAP achieved: {best_mAP:.4f}", level=1)
                print('current mAP: {:5.1%}  best mAP: {:5.1%}'.format(mAP, best_mAP))
        else:
            log(f"Epoch {epoch + 1}/{args.epochs} - skipping evaluation", level=2)

    torch.save(model.state_dict(), osp.join(args.logs_dir, 'last.pth'))
    log("Saved final model checkpoint", level=1)

    # results
    model.load_state_dict(torch.load(osp.join(args.logs_dir, 'best.pth')))
    log("Loaded best model for final evaluation", level=2)
    with torch.no_grad():
        final_results = evaluator.evaluate(test_loader, test_dataset.query, test_dataset.gallery, cmc_flag=True)
    if wandb is not None and getattr(wandb, "run", None) is not None:
        try:
            log_dict = {}
            cmc_scores = None
            final_mAP = None
            final_visfig = None

            if isinstance(final_results, tuple):
                if len(final_results) >= 3:
                    cmc_scores, final_mAP, final_visfig = final_results[0], final_results[1], final_results[2]
                elif len(final_results) == 2:
                    final_mAP, final_visfig = final_results
            else:
                final_mAP = final_results

            if final_mAP is not None:
                log_dict['final/mAP'] = final_mAP
            if cmc_scores is not None:
                for rank, score in enumerate(cmc_scores, start=1):
                    if rank in [1, 5, 10]:
                        log_dict[f'eval/CMC_top_{rank}'] = score
            if final_visfig is not None:
                final_fig = getattr(final_visfig, 'figure', None)
                if final_fig is not None:
                    final_vis_path = osp.join(args.logs_dir, 'vis_final.png')
                    final_fig.savefig(final_vis_path, bbox_inches='tight')
                    plt.close(final_fig)
                    log_dict['final/embedding_vis'] = wandb.Image(final_vis_path, caption='Best Model Embedding')

            if log_dict:
                wandb.log(log_dict)
            wandb.finish()
        except Exception:
            pass
    log("Completed final evaluation", level=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Balancing Alignment and Uniformity (BAU)")
    
    # data
    parser.add_argument('-ds', '--source-dataset', nargs='+', type=str, default=['msmt17', 'market1501', 'cuhksysu'])
    parser.add_argument('-dt', '--target-dataset', type=str, default='cuhk03')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-n', '--num-instances', type=int, default=4,
                        help='each minibatch consist of '
                             '(batch_size // num_instances) identities, and '
                             'each identity has num_instances instances, '
                             'default: 0 (NOT USE)')
    parser.add_argument('--height', type=int, default=256, help='input height')
    parser.add_argument('--width', type=int, default=128, help='input width')

    # path  
    working_dir = osp.dirname(osp.abspath(__file__))
    # parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default="/home/stud/leez/storage/user/reid/data")
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs/test'))

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-epochs', nargs='+', type=int, default=[50,55])

    # train
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.names())
    parser.add_argument('--margin', type=float, default=0.3, help='margin parameter for triplet loss')
    parser.add_argument('--lam', type=float, default=1.5, help='weighting parameter for alignment loss')
    parser.add_argument('--k', type=int, default=10, help='k-NN parameter for weighting strategy')
    parser.add_argument('--prob', type=float, default=0.5, help='probability of applying data augmentation to inputs')

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=60)           
    parser.add_argument('--iters', type=int, default=500, help='iteration for each epoch')
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30, 50], help='milestones for the learning rate decay')

    # ablation
    parser.add_argument('--use-aug-ce', action='store_true', help='use cross entropy loss on strongly augmented images')
    parser.add_argument('--no-align', action='store_true', help='disable alignment loss')
    parser.add_argument('--no-uniform', action='store_true', help='disable uniformity loss')
    parser.add_argument('--no-domain', action='store_true', help='disable domain loss')
    parser.add_argument('--no-triplet', action='store_true', help='disable triplet loss')
    parser.add_argument('--no-ce', action='store_true', help='disable cross-entropy loss')
    
    # manifold-aware
    parser.add_argument('--manifold-aware', type=lambda x: x.lower() == 'true', default=False, help='use manifold-aware distance computations (poincare ball)') 
    parser.add_argument('--curvature', type=float, default=0.0, help='manifold curvature (only used if manifold-aware is set)')
    parser.add_argument('--manifold-chunk-size', type=parse_optional_chunk_size, default=None,
                        help="chunk size for manifold distance computation; set to 'none' to disable chunking")

    # finsler manifolds (Rander's metric)
    parser.add_argument('--omega', type=float, default=None, help='omega parameter for Finsler manifold (Rander metric)')
    
    # fine-tuning
    parser.add_argument('--fine-tuning', type=lambda x: x.lower() == 'true', default=False, help='fine-tune the model')
    parser.add_argument('--checkpoint-path', type=str, default='', help='path to the checkpoint to load')

    parser.add_argument('--wandb-name', type=str, default='', help='additional name info for wandb runs')
    parser.add_argument('--verbosity', type=int, default=0, choices=[0, 1, 2], help='increase output verbosity: 0=minimal, 1=info, 2=debug')
    main()

