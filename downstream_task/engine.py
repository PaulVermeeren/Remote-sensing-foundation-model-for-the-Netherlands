import math
import os
import sys
import torch
import torch.nn as nn
from typing import Iterable
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from loss import calculate_loss

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from util import misc
from util import lr_sched

def train_one_epoch(model: nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, rank: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch+1}] GPU: [{rank}]'
    print_freq = 200
    
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, (samples, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples = samples.to(device, non_blocking=True)
        
        if args.dataset == "dior":
            new_target = []
            for t in target:
                target_box = t[0].to(device, non_blocking=True)
                target_class = t[1].to(device, non_blocking=True)
                new_target.append((target_box, target_class))
            target = new_target
        else:
            target = target.to(device, non_blocking=True)
                                
        with torch.amp.autocast('cuda'):
            outputs = model(samples)  
            loss = calculate_loss(args, outputs, target)
        
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        
    torch.cuda.empty_cache()

    metric_logger.synchronize_between_processes()
    if rank == 0:
        print(f"Averaged loss across all GPUs: {metric_logger.meters['loss'].global_avg}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}