# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

#assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models_rip import MultiAV_RIP,MultiAV2, load_checkpoint
from collections import defaultdict
from engine_finetune import train_one_epoch, evaluate
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch.nn as nn
import torch.nn.functional as F
class WeightedLabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing and class weights.
    """
    def __init__(self, smoothing=0.1, weight=None):
        super(WeightedLabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.weight = weight

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        if self.weight is not None:
            loss *= self.weight[target]
        
        return loss.mean()
# class WeightedLabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self, smoothing=0.1, weight=None):
#         super(WeightedLabelSmoothingCrossEntropy, self).__init__()
#         self.smoothing = smoothing
#         self.weight = weight
#         if weight is not None:
#             self.weight = torch.tensor(weight, dtype=torch.float32)
#         else:
#             self.weight = None

#     def forward(self, input, target):
#         log_prob = F.log_softmax(input, dim=-1)
#         weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
#         weight.scatter_(-1, target.unsqueeze(-1), 1 - self.smoothing)
#         if self.weight is not None:
#             weight = weight * self.weight[target].view(-1, 1)
        
#         loss = (-weight * log_prob).sum(dim=-1).mean()
#         return loss

def saferound(x, digits=0):
    """
    四舍五入到指定的小数位数，同时确保四舍五入后的值之和与原始值的总和相同。
    
    :param x: 要四舍五入的数值或数值数组。
    :param digits: 要四舍五入到的小数位数，默认为0，即四舍五入到整数。
    :return: 四舍五入后的数值或数值数组。
    """
    # 计算四舍五入后的值
    rounded = np.round(x - 0.5e-10, decimals=digits).astype(int)
    # 计算四舍五入的误差
    error = x - rounded
    # 如果误差的和不等于0，调整最后一个值
    if not np.isclose(error.sum(), 0):
        diff = int(round(error.sum()))
        rounded[-1] += diff
    return rounded

from torch.utils.data import Sampler
from operator import itemgetter
class WeightedBalanceClassSampler(Sampler):
    """Allows you to create stratified sample on unbalanced classes with given probabilities (weights).
    Args:
        labels: list of class label for each elem in the dataset
        weight: A sequence of weights to balance classes, not necessary summing up to one.
        length: the length of the sample dataset.
    """

    def __init__(
        self, labels, weight, length,
    ):
        """Sampler initialisation."""
        super().__init__(labels)

        labels = np.array(labels).astype(np.int32)

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }
        weight = np.array(weight)
        weight = weight / weight.sum()

        samples_per_class = weight * length

        samples_per_class = np.array(saferound(samples_per_class, 0)).astype(np.int32)

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = length

    def __iter__(self):
        """
        Yields:
            indices of stratified sample
        """
        indices = []
        for key in sorted(self.lbl2idx):
            replace_flag = self.samples_per_class[key] > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class[key], replace=replace_flag
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length
class DatasetFromSampler(torch.utils.data.Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self):
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)
class DistributedSamplerWrapper(torch.utils.data.DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler=None,
        num_replicas=1,
        rank=0,
        shuffle=True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))
def set_requires_grad(nets, requires_grad=True):
    for name,param in nets.named_parameters():
        
        if 'sn_unet' in name:
            param.requires_grad = requires_grad
    for name,param in nets.named_parameters():
        print(f'{name}: {param.requires_grad}') 
    return nets
def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.9,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=5e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.2, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.1,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.1,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--task', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    
    
    parser.add_argument('--cls_weight', action='store_true',
                        help='Perform cls_weight only')
    # Dataset parameters
    parser.add_argument('--data_path', default='/home/jupyter/Mor_DR_data/data/data/IDRID/Disease_Grading/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    print(f'seed is: {seed}')
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
    dataset_train = build_dataset(is_train='train', args=args)
    dataset_val = build_dataset(is_train='val', args=args)
    dataset_test = build_dataset(is_train='test', args=args)

    if args.cls_weight:
        # 计算每个类别的样本数量
        class_counts = defaultdict(int)
        label_list = []
        for _, label in dataset_train:
            class_counts[label] += 1
            label_list.append(label)
        # 计算总样本数量
        total_samples = len(dataset_train)

        # 计算每个类别的比例
        class_proportions = {cls: count / total_samples for cls, count in class_counts.items()}

       

        # 根据比例计算权重（例如，使用逆比例作为权重）
        class_weights = {cls: 1.0 / prop for cls, prop in class_proportions.items()}
        max_weight = max(class_weights.values())
        min_weight = min(class_weights.values())
        max_allowed_weight = min_weight * 10

        max_allowed_weight = min_weight * 10
        adjusted_class_weights = {cls: min(weight, max_allowed_weight) for cls, weight in class_weights.items()}
        total_adjusted_weights = sum(adjusted_class_weights.values())
        final_class_weights = {cls: weight / total_adjusted_weights for cls, weight in adjusted_class_weights.items()}

        # total_weights = sum(class_weights.values())
        # class_weights = {cls: weight / total_weights for cls, weight in class_weights.items()}
        # 转换为PyTorch张量
        weights_tensor = torch.tensor([final_class_weights[cls] for cls in range(len(class_counts))], dtype=torch.float32).cuda()

        #imblanceSampler = WeightedBalanceClassSampler(labels=label_list,weight=weights_tensor,length=total_samples)


        print(f'========================weight: {weights_tensor}==============')

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        # if args.cls_weight:
        #     sampler_train = DistributedSamplerWrapper(imblanceSampler,num_replicas=num_tasks, rank=global_rank, shuffle=True)
        # else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            
        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
            
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir+args.task)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        
        drop_last=False
    )
    
    
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    model = MultiAV_RIP(num_classes=args.nb_classes,pretrain=False,drop_path=args.drop_path)
    #model = MultiAV2(num_classes=args.nb_classes,pretrain=True)
    if not args.eval:
        
        model = load_checkpoint(model,args.finetune)

        #trunc_normal_(model.head[2].weight, std=2e-5)

    model.to(device)
    #model = set_requires_grad(model,requires_grad=False)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr*eff_batch_size/256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd_rip(model_without_ddp, args.weight_decay,
        
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
        #criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    elif args.smoothing > 0.:
        if args.cls_weight:
            criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor,label_smoothing=args.smoothing)
        else:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        if args.cls_weight:
            criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats,auc_roc,_,_ = evaluate(data_loader_test, model, device, args.task, epoch=0, mode='test',num_class=args.nb_classes)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.5
    max_auc = 0.0
    max_pr=0.0
    last_pr_auc = 0.0
    last_acc_auc = 0.0
    last_pr = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        val_stats,val_auc_roc, val_auc_pr,val_acc = evaluate(data_loader_val, model, device,args.task,epoch, mode='val',num_class=args.nb_classes)
        # if  (max_accuracy<val_acc) and (max_auc<val_auc_roc):
        #     #max_auc = max(val_auc_roc,max_auc)
        #     max_accuracy = max(max_accuracy,val_acc)
        #     max_auc = max(val_auc_roc,max_auc)
            
            
        #     if args.output_dir :
        #         misc.save_model(
        #             args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #             loss_scaler=loss_scaler, epoch='best_all')
        #         test_stats,auc_roc,auc_pr,acc = evaluate(data_loader_test, model, device,args.task,epoch, mode='test',num_class=args.nb_classes)
        
        #         print(f'Max ALL: {max_accuracy:.2f}%   Test: {acc:.2f}%')
        #if (last_pr<val_auc_pr and max_auc==val_auc_roc) or  (max_auc<val_auc_roc):
        if (max_auc<val_auc_roc):
            max_auc = max(val_auc_roc,max_auc)
            #max_pr = max(val_auc_pr,max_pr)
            last_pr = val_auc_pr
            
            if args.output_dir :
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch='best_auc')
                test_stats,auc_roc,auc_pr,acc = evaluate(data_loader_test, model, device,args.task,epoch, mode='test',num_class=args.nb_classes)
        
                print(f'Max AUC: {max_auc:.2f}%   Test: {auc_roc:.2f}%')

        #if (last_pr_auc<val_auc_roc and max_pr==val_auc_pr) or  (max_pr<val_auc_pr):
        if (max_pr<val_auc_pr):    
            #max_auc = max(val_auc_roc,max_auc)
            max_pr = max(val_auc_pr,max_pr)
            last_pr_auc = val_auc_roc
                
            if args.output_dir :
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch='best_pr')
                test_stats,auc_roc,auc_pr,acc = evaluate(data_loader_test, model, device,args.task,epoch, mode='test',num_class=args.nb_classes)
        
                print(f'Max PR: {max_pr:.2f}%   Test: {auc_pr:.2f}%')
        
        #if (last_acc_auc<val_auc_roc and max_accuracy==val_acc) or  (max_accuracy<val_acc):
        if (max_accuracy<val_acc):
            #max_auc = max(val_auc_roc,max_auc)
            max_accuracy = max(max_accuracy,val_acc)
            
            last_acc_auc = val_auc_roc
            
            if args.output_dir :
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch='best_acc')
                test_stats,auc_roc,auc_pr,acc = evaluate(data_loader_test, model, device,args.task,epoch, mode='test',num_class=args.nb_classes)
        
                print(f'Max ACC: {max_accuracy:.2f}%   Test: {acc:.2f}%')
        
        
        # if args.nb_classes==2 and max_accuracy<=val_stats['acc1']:
        # #if args.nb_classes==2 and max_auc<val_auc_roc:
        #     if max_accuracy<val_stats['acc1']:
        #         max_accuracy=val_stats['acc1']
        #         max_auc=max(val_auc_roc,max_auc) 
        #         if args.output_dir:
        #             misc.save_model(
        #             args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #             loss_scaler=loss_scaler, epoch=epoch)
                
        #             test_stats,auc_roc,_ = evaluate(data_loader_test, model, device,args.task,epoch, mode='test',num_class=args.nb_classes)
        #     else:
        #         if max_auc<=val_auc_roc:
        #             max_auc=max(val_auc_roc,max_auc) 
        #             if args.output_dir:
        #                 misc.save_model(
        #                 args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #                 loss_scaler=loss_scaler, epoch=epoch)
                
        #                 test_stats,auc_roc,_ = evaluate(data_loader_test, model, device,args.task,epoch, mode='test',num_class=args.nb_classes)


        if epoch==(args.epochs-1):
            test_stats,auc_roc,_,_ = evaluate(data_loader_test, model, device,args.task,epoch, mode='test',num_class=args.nb_classes)
            if args.output_dir:
                    misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        
        if log_writer is not None:
            log_writer.add_scalar('perf/val_acc1', val_stats['acc1'], epoch)
            log_writer.add_scalar('perf/val_auc', val_auc_roc, epoch)
            log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    with open(os.path.join(args.output_dir, "time.txt"), mode="w", encoding="utf-8") as f:
                f.write(f'Training time {total_time_str}')
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
