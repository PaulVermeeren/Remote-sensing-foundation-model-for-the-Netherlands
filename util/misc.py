# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import glob
import time
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

import torch
import torch.distributed as dist
from torch import inf
import shutil
#from util.pos_embed import interpolate_pos_embed

import torch.nn.functional as F
import torch.nn as nn
import random
import string
import gc

import socket
from contextlib import closing

def get_open_port():
    """
    Find available port for DDP
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
        
def setup(args, rank, world_size):
    """
    Setting environment for DDP
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """
    Cleaning up DDP
    """
    torch.distributed.barrier()
    dist.destroy_process_group()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

def signal_handler(sig, frame):
    print("Ctrl + C detected. Stopping training.")
    torch.distributed.barrier()
    dist.destroy_process_group()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    sys.exit(0)

def generate_unique_id():
    """
    Generating random unique id for saving the run
    """
    while True:
        random_id = ''.join(random.choices(string.ascii_letters, k=4))
        if not any(f"{random_id}" in filename for filename in os.listdir('output_dir')):
            return random_id    

class EarlyStopping:
    """
    Early stopping for both training loss and validation measures
    patience: maximum non-improving runs
    delta: minimum improvement, 1.2 implies 20% better
    verbose: display count
    val_tag: tag for validation measure to distinquish the prints
    """
    def __init__(self, patience=50, delta=1, verbose=False, val_tag="", minim=True):
        self.patience = patience
        self.counter = 0
        if minim:
            self.best_loss = np.inf
        else:
            self.best_loss = 0
        self.delta = delta
        self.verbose = verbose
        self.early_stop = False
        self.is_best = False
        self.minim = minim
        self.val_tag = val_tag

    def __call__(self, loss):
        self.is_best = False
        if (self.minim and loss * self.delta < self.best_loss) or (not self.minim and loss > self.best_loss* self.delta):
            self.best_loss = loss
            self.counter = 0
            self.is_best = True
        else: 
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  
            elif self.verbose:
                if self.val_tag != "":
                    print(f"Counter {self.val_tag}: ",self.counter)
                    print(f"Best score {self.val_tag}: ",self.best_loss)
                else:
                    print("Counter: ",self.counter)
                    print("Best loss: ",self.best_loss)





################################
#      original functions      #
################################




class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB), training_loop=True)
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)), training_loop=True)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_printing(is_master, disable_logging):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        training_loop = kwargs.pop('training_loop', False)
        
        if (is_master and not disable_logging) or force or training_loop:
            now = datetime.datetime.now().time()
            #builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


class MaskPaddedGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask):
        ctx.save_for_backward(mask)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        m = grad_output * mask
        #print(m.numel())
        #print(torch.count_nonzero(m))
        return m, None

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, mask=None):
        #self._scaler.scale(loss).backward(create_graph=create_graph)
        scaled_loss = self._scaler.scale(loss)
        if mask is not None:
            masked_loss = MaskPaddedGradients.apply(scaled_loss, mask)
            masked_loss.backward(create_graph=create_graph)
        else:
            scaled_loss.backward(create_graph=create_graph)
            
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, rand_id, val_tag=""):
    if loss_scaler is not None:
        output_dir = Path(args.output_dir)
        today = datetime.datetime.today().strftime('%Y-%m-%d')
    
        # prepend task name
        if hasattr(args, 'dataset'):
            today = f'{args.dataset}_{today}'
            
        if val_tag != "":
            epoch_name = f"{today}_{args.tag}_{epoch}_{args.rand_id}_{val_tag}"
            pattern = os.path.join(output_dir, f"*_{args.rand_id}_{val_tag}.pth")
        else:
            epoch_name = f"{today}_{args.tag}_{epoch}_{args.rand_id}"
            pattern = os.path.join(output_dir, f"*_{args.rand_id}.pth")
            
        best_checkpoint = os.path.join(output_dir, f"{epoch_name}.pth")
    
        # Remove all previous checkpoints that start with today's date and rand_id
        for checkpoint_path in glob.glob(pattern):
            os.remove(checkpoint_path)
            
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }
    
        save_on_master(to_save, best_checkpoint)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="nolossscalar_checkpoint-%s" % epoch_name, client_state=client_state)

def adapt_channels(state_dict, new_in_channels):

    # Get the current patch_embed.proj.weight
    old_weight = state_dict['model.patch_embed.proj.weight']
    old_in_channels = old_weight.shape[1]
    out_channels, _, kernel_size, _ = old_weight.shape

    # Create a new weight tensor with the desired number of input channels
    new_weight = torch.zeros(out_channels, new_in_channels, kernel_size, kernel_size)

    # Copy existing channels
    min_channels = min(old_in_channels, new_in_channels)
    new_weight[:, :min_channels, :, :] = old_weight[:, :min_channels, :, :]

    # If we're adding channels, initialize them randomly
    if new_in_channels > old_in_channels:
        nn.init.kaiming_normal_(new_weight[:, old_in_channels:, :, :])

        # Scale the new channels to match the magnitude of the existing ones
        existing_magnitude = old_weight.norm(p=2, dim=(1, 2, 3), keepdim=True)
        new_channels_magnitude = new_weight[:, old_in_channels:, :, :].norm(p=2, dim=(1, 2, 3), keepdim=True)
        scaling_factor = existing_magnitude / new_channels_magnitude
        new_weight[:, old_in_channels:, :, :] *= scaling_factor

    # If we're reducing channels, we might want to rescale to maintain overall magnitude
    elif new_in_channels < old_in_channels:
        scale_factor = (old_in_channels / new_in_channels) ** 0.5
        new_weight *= scale_factor

    # Update the state dict with the new weight
    state_dict['model.patch_embed.proj.weight'] = new_weight

    return state_dict

def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            # del checkpoint['model']['head.weight']
            # del checkpoint['model']['head.bias']
            
        checkpoint_model = checkpoint['model']
        #checkpoint_model = adapt_channels(checkpoint_model, args.in_chans) # als andere channels moeten bij resume head, gebruikt voor C=4 naar C=6 andere dataset
        model_without_ddp.load_state_dict(checkpoint_model, strict=False)
        if dist.get_rank() == 0:
            print("Resume checkpoint %s" % args.resume)
                
        if 'args' in checkpoint and args.continue_params:
        
            if args.transfer_learning and 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
                optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                    
                if dist.get_rank() == 0:                
                    print("With optim & sched!")
                    
            # Update learning rate related arguments
            checkpoint_args = checkpoint['args']
            args.min_lr = (checkpoint_args.min_lr / checkpoint_args.world_size) * args.world_size
            args.lr = (checkpoint_args.lr / checkpoint_args.world_size) * args.world_size
            args.blr = checkpoint_args.blr
            args.warmup_epochs = checkpoint_args.warmup_epochs
            args.epochs = checkpoint_args.epochs
            if hasattr(args, 'transfer_learning'):
                if args.transfer_learning:
                    args.start_epoch = checkpoint['epoch'] + 1
            
            if dist.get_rank() == 0:
                print("Loaded args")
            
            # vegetatie
            if hasattr(checkpoint_args, 'cross') and hasattr(args, 'cross'):
                args.cross = checkpoint_args.cross
            if hasattr(args, 'v'):
                args.v = checkpoint_args.v
            if hasattr(args, 'cnnhead'):
                args.cnnhead = checkpoint_args.cnnhead
                
            if dist.get_rank() == 0 and hasattr(checkpoint_args, 'tensorboard_log_dir') and hasattr(args, 'transfer_learning'):
                if args.transfer_learning:
                    original_log_dir = checkpoint_args.tensorboard_log_dir 
            
                    new_log_dir = f"{args.log_dir}/{args.dataset}_{datetime.datetime.now().strftime('%Y-%m-%d')}_{args.tag}_{args.rand_id}/"
                    
                    shutil.copytree(original_log_dir, new_log_dir)
                    print(f"Copied original logs from {original_log_dir} to {new_log_dir}")
                
            
        
            #print(f"Resuming TensorBoard logging in {args.tensorboard_log_dir}")

def load_model_different_size(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint_model = checkpoint['model']
            state_dict = model_without_ddp.state_dict()
            # del checkpoint['model']['head.weight']
            # del checkpoint['model']['head.bias']
        for k in ['pos_embed', 'decoder_pos_embed','patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
            if k in checkpoint['model'] and checkpoint['model'][k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint['model'][k]

        # interpolate position embedding
        #interpolate_pos_embed(model_without_ddp, checkpoint['model'])

        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
