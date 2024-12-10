# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import ModelEma,accuracy
from loss_sim import NegativeMiningInfoNCECriterion
import utils
import torch.nn.functional as F

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, criterion2=None,log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples,sample_single, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples['positive'][0] = samples['positive'][0].to(device, non_blocking=True)
        samples['positive'][1] = samples['positive'][1].to(device, non_blocking=True)
        samples['negative'][0] = samples['negative'][0].to(device, non_blocking=True)
        samples['negative'][1] = samples['negative'][1].to(device, non_blocking=True)
        sample_single = sample_single.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            sample_single, targets = mixup_fn(sample_single, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output,output_single = model(samples,sample_single)
                loss1 = criterion(output)
                loss2 = criterion2(output_single,targets)
        else: # full precision
            output,output_single = model(samples,sample_single)
            loss1 = criterion(output)
            loss2 = criterion2(output_single,targets)
        loss = loss1+0.1*loss2
        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            #output = F.Softmax(output)
            acc1 = accuracy(output_single, targets)[0]
            batch_size = output_single.shape[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            class_acc = loss_value
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processesc
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print(f'Train Metrics - Acc@1 {metric_logger.acc1.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f} ')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(data_loader, model, device,task='./',mode='val', use_amp=False,num_class=2):
    print(f'num_class:=================={num_class}')
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if mode=='val':
        header = 'Eval:'
    else:
        header='Test'
    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    # switch to evaluation mode
    model.eval()
    #for data_iter_step, (samples,sample_single, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
    for batch in metric_logger.log_every(data_loader, 10, header):
        
        samples = batch[0]
        sample_single=batch[1]
        target = batch[-1]
        
        samples['positive'][0] = samples['positive'][0].to(device, non_blocking=True)
        samples['positive'][1] = samples['positive'][1].to(device, non_blocking=True)
        samples['negative'][0] = samples['negative'][0].to(device, non_blocking=True)
        samples['negative'][1] = samples['negative'][1].to(device, non_blocking=True)
        sample_single = sample_single.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)
        
        
        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output,output_single = model(samples,sample_single)
                loss = criterion(output_single, target)
        else:
            output,output_single = model(samples,sample_single)
            loss = criterion(output_single, target)
        

        acc1 = accuracy(output_single, target)[0]
        batch_size = output_single.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes

    metric_logger.synchronize_between_processes()

    print(f'{mode} Metrics - Acc@1 {metric_logger.acc1.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f} ')
    
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}