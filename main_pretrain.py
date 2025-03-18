import sys
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7,8"
os.environ["PYTHONWARNINGS"] = "ignore:Failed to load image Python extension:"

sys.path.append('model')

import argparse
import datetime
import json
import numpy as np
import time
import random
from pathlib import Path
import warnings
import string
import tifffile as tiff
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import pandas as pd

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
#assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
from util.misc import setup_printing

from engine_pretrain import train_one_epoch
import model.swin_mae as swin_mae

from itertools import islice

warnings.filterwarnings("ignore")

def setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

class CustomDataLoader(Dataset):
    def __init__(self, path, augmentation=1):
        self.path = path
        self.augmentation = augmentation
        self.samples = self.make_dataset(path)

    def make_dataset(self, filepath):
        samples = []
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)

            for i, key in enumerate(data):
                temporal_list = data[key]
                paths = []
                for datapoint in temporal_list:
                    if datapoint != {}:
                        filename = os.path.basename(datapoint.get('uri'))
                        path = '../data/pretrain/'+filename
                        paths.append(path)

                if len(paths) == 6:
                    samples.append((paths, False))

                    for _ in range(self.augmentation):
                        samples.append((paths, True))

        print(f"Length samples {len(samples)}")
        print(f"Length unique samples {len(samples)/(self.augmentation+1)}")
        return samples

    def __getitem__(self, index):
        paths, augment = self.samples[index]
        imgs = []
        for path in paths:
            img = tiff.imread(path)

            # pad for 4 channels
            if img.shape[2] == 4:
                padded_image = np.zeros((img.shape[0], img.shape[1], 6), dtype=img.dtype)
                padded_image[:, :, :4] = img
                img = padded_image

            # Normalize using percentiles
            min_val = np.percentile(img, 2, axis=(0, 1))
            max_val = np.percentile(img, 98, axis=(0, 1))
            img = np.clip(img, min_val, max_val)
            img = (img - min_val) / (max_val - min_val + 1e-06)

            # (H, W, C) to (C, H, W)
            img = np.transpose(img, (2, 0, 1))

            img = torch.tensor(img, dtype=torch.float32)

            imgs.append(img)

        imgs = torch.stack(imgs, dim=0)

        if augment:
            imgs = self.apply_augmentations(imgs)

        return imgs


    def random_crop(self, img):
        _, _, h, w = img.shape
        crop_size = random.randint(w // 2, w)  # Random crop size
        top = torch.randint(0, h - crop_size + 1, (1,)).item()
        left = torch.randint(0, w - crop_size + 1, (1,)).item()
        img_cropped = img[:, :, top:top + crop_size, left:left + crop_size]
        img_resized = F.interpolate(img_cropped, size=(w, w), mode='bilinear', align_corners=False)
        return img_resized

    def random_rotation(self, img):
        rotation = random.choice([0, 90, 180, 270])
        img_rotated = TF.rotate(img, rotation)
        return img_rotated

    def random_flip(self, img):
        if random.random() > 0.5:
            img = TF.hflip(img)
        else:
            img = TF.vflip(img)
        return img

    def random_color_distortion(self, img):
        # Apply brightness adjustment
        brightness_factor = random.uniform(0.8, 1.2)
        img = img * brightness_factor
        img = torch.clamp(img, 0.0, 1.0)  # Ensure values remain valid

        # Apply contrast adjustment
        mean_per_channel = img.mean(dim=(2, 3), keepdim=True)  # Compute mean per channel
        contrast_factor = random.uniform(0.8, 1.2)
        img = (img - mean_per_channel) * contrast_factor + mean_per_channel
        img = torch.clamp(img, 0.0, 1.0)

        # Apply saturation adjustment
        # Convert to grayscale-like by averaging over spatial dimensions, and adjust intensity
        saturation_factor = random.uniform(0.8, 1.2)
        img_mean = img.mean(dim=1, keepdim=True)  # Compute the mean intensity across channels
        img = (img - img_mean) * saturation_factor + img_mean
        img = torch.clamp(img, 0.0, 1.0)
        return img

    def random_blur(self, img):
        # Apply Gaussian blur
        kernel_size = random.choice([3, 5])  # Kernel size for blur
        img = TF.gaussian_blur(img, kernel_size=kernel_size)
        return img

    def random_noise(self, img):
        # Add random Gaussian noise
        noise = torch.randn_like(img) * 0.05  # Noise scale
        img = img + noise
        img = torch.clamp(img, 0.0, 1.0)  # Ensure values remain valid
        return img

    def apply_augmentations(self, img):
        # Group 1: Spatial augmentations
        spatial_augmentations = [self.random_crop, self.random_rotation, self.random_flip]
        chosen_spatial_aug = random.choice(spatial_augmentations)
        img = chosen_spatial_aug(img)

        # Group 2: Intensity augmentations
        intensity_augmentations = [self.random_color_distortion, self.random_blur, self.random_noise]
        chosen_intensity_aug = random.choice(intensity_augmentations)
        img = chosen_intensity_aug(img)

        return img


    def __len__(self):
        return len(self.samples)


class EarlyStopping:
    def __init__(self, patience=20, delta=1, verbose=False):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.delta = delta
        self.verbose = verbose
        self.early_stop = False
        self.is_best = False

    def __call__(self, loss):
        self.is_best = False
        if loss * self.delta < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            self.is_best = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            elif self.verbose:
                print("Counter: ",self.counter)
                print("Best loss: ",self.best_loss)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--ip', default=12373, type=int,
                        help='')

    # Model parameters
    parser.add_argument('--patch_size', default=4, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.9, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--img_size', default=512, type=int,
                        help='')
    parser.add_argument('--in_chans', default=6, type=int,
                        help='')
    parser.add_argument('--window_size', default=8, type=int,
                        help='')
    parser.add_argument('--depths', default=(2, 2, 6, 2), type=tuple,
                        help='')
    parser.add_argument('--num_heads', default=(3, 6, 12, 24), type=tuple,
                        help='')
    parser.add_argument('--mask_size', default=4, type=int,
                        help='')
    parser.add_argument('--temporal_size', default=6, type=int,
                        help='')
    parser.add_argument('--temporal_bool', action='store_false',
                        help='')
    parser.add_argument('--pimask', action='store_false',
                        help='')
    parser.add_argument('--high_pass', action='store_false',
                        help='')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.001, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    #provincie_utrecht2_succes
    parser.add_argument('--train_path', default='../data/download/merged_pretrain_download.json', type=str,
                        help='Train.csv path')
    parser.add_argument('--val_path', default='')
    parser.add_argument('--tag', default='', type=str,
                        help='tag')
    parser.add_argument('--tag_long', default='', type=str,
                        help='tag long')

    parser.add_argument('--output_dir', default='output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='logs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--disable_logging', action='store_true',
                        help='Disable all logging for testing purposes')

    return parser


def validate(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for images in data_loader:
            images = images.to(device, non_blocking=True)
            loss, _, _, _ = model(images)
            total_loss += loss.item()
            total_samples += 1

    total_loss = torch.tensor(total_loss).to(device)
    total_samples = torch.tensor(total_samples).to(device)
    dist.all_reduce(total_loss)
    dist.all_reduce(total_samples)
    total_loss = total_loss.item()
    total_samples = total_samples.item()

    return total_loss / total_samples

def main(rank, args):
    setup(args, rank, args.world_size)

    # Fixed random seeds
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set up training equipment
    torch.cuda.set_device(rank)
    device = torch.device(args.device)
    cudnn.benchmark = True

    setup_printing(rank==0, args.disable_logging)


    # logger
    if rank == 0 and args.log_dir is not None and not args.disable_logging:
        os.makedirs(args.log_dir, exist_ok=True)
        print(args.tag)
        path = f"{args.log_dir}/{datetime.datetime.now().strftime('%Y-%m-%d')}_{args.tag}_{args.rand_id}/"
        log_writer = SummaryWriter(log_dir=path)

        # add args
        args_dict = vars(args)
        args_str = "\n\n".join([f"{k}: {v}" for k, v in args_dict.items()])
        log_writer.add_text('arguments', args_str, global_step=0)

        log_writer.flush()
    else:
        log_writer = None

    # create dataset and loader
    dataset_train = CustomDataLoader(args.train_path, 1)
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=args.world_size, rank=rank, shuffle=True)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=False, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    #dataset_val = CustomDataLoader(args.val_path, 0)
    #sampler_train_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=args.world_size, rank=rank, shuffle=True)
    #data_loader_val = torch.utils.data.DataLoader(
    #    dataset_val,
    #    shuffle=False, sampler=sampler_train_val,
    #    batch_size=args.batch_size,
    #    num_workers=1,
    #    pin_memory=True,
    #    drop_last=True,
    #    persistent_workers=False
    #)

    # define model
    model = swin_mae.__dict__['swin_mae'](img_size=args.img_size, patch_size=args.patch_size, in_chans=args.in_chans, depths=args.depths, num_heads=args.num_heads, window_size=args.window_size, mask_ratio=args.mask_ratio, mask_size=args.mask_size, temporal_bool=args.temporal_bool, pimask=args.pimask, temporal_size=args.temporal_size,high_pass=args.high_pass)

    model_without_ddp = model
    model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    args.total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters {args.total_parameters}")

    # only base_lr is specified
    eff_batch_size = args.batch_size * args.accum_iter * args.world_size * args.temporal_size
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / args.img_size
    args.min_lr = args.lr * 0.01

    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if rank == 0:
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()

    early_stopping = EarlyStopping(verbose=(True if rank == 0 else False))
    early_stopping_val = EarlyStopping(verbose=(True if rank == 0 else False))

    epochs_trained = 0
    for epoch in range(args.start_epoch, args.epochs):
        epochs_trained = epoch

        data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, rank, loss_scaler,
            log_writer=log_writer,
            args=args,
        )

        if rank == 0 and log_writer is not None:
            log_writer.add_scalar('average_train/loss', train_stats['loss'], epoch)

        # intermediate saving and last epoch saving
        #if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            #val_loss = validate(model, data_loader_val, device)
            #if rank == 0:
                #print(f"Validation Loss: {val_loss:.4f}")
                #if log_writer is not None:
                #    log_writer.add_scalar('val/loss', val_loss, epoch)
                #
                #early_stopping_val(val_loss)
                #if early_stopping_val.is_best: ## save val file
                #    misc.save_model(
                #        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                #        loss_scaler=loss_scaler, epoch=epoch+1, rand_id = args.rand_id, val=True)

                #if early_stopping.counter == 0: ## save regular file
                #misc.save_model(
                #    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                #    loss_scaler=loss_scaler, epoch=epoch+1, rand_id = args.rand_id)
                #print(f"Saving at {epoch+1} with id {args.rand_id}")
        if rank == 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch+1, rand_id = args.rand_id)
            print(f"Saving at {epoch+1} with id {args.rand_id}")

        if not args.disable_logging:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, }

        if args.output_dir and rank == 0 and not args.disable_logging:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        #early_stopping(train_stats['loss'])
        #if rank == 0 and early_stopping.is_best:
        #    misc.save_model(
        #        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #        loss_scaler=loss_scaler, epoch=epoch+1, rand_id = args.rand_id)
        #    print(f"Saving at epoch {epoch+1} with id {args.rand_id}")

        #if rank == 0 and early_stopping.early_stop:
        #    print("Early stopping")
        #    break

    if rank == 0:
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        final_stats = {
            'Epochs Trained': epochs_trained,
            'Final Training Loss': train_stats['loss'],
            'Total Training Time': total_time_str,
        }

        args_str = "\n\n".join([f"{k}: {v}" for k, v in final_stats.items()])
        log_writer.add_text('Final Statistics', args_str, global_step=1)

        log_writer.close()

    cleanup()


def generate_unique_id():
    while True:
        random_id = ''.join(random.choices(string.ascii_letters, k=4))
        if not any(f"{random_id}" in filename for filename in os.listdir('output_dir')):
            return random_id


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.world_size = torch.cuda.device_count()
    args.rand_id = generate_unique_id()
    mp.spawn(main, nprocs=args.world_size, args=(args,), join=True)