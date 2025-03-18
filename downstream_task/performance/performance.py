import sys
import os
import argparse
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.classification_validation import validate as classification_validation
from validation.segmentation_validation import validate as segmentation_validation
from validation.change_validation import validate as change_validation
from validation.height_validation import validate as height_validation

from loader import CustomDataLoader
from validate import ValidationHandler
from misc import get_args_parser, create_model
from config import config

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.misc import setup, cleanup


def main(rank, args):
    # setup cuda
    setup(args, rank, args.world_size)
    torch.cuda.set_device(rank)
    device = torch.device(args.device)
    cudnn.benchmark = True

    # Dataset
    dataset_test = CustomDataLoader(args, path="../"+args.test_path, num_augmentations=0, prepend="../")
    sampler = DistributedSampler(dataset_test, num_replicas=args.world_size, rank=rank, shuffle=False)
    data_loader = DataLoader(
        dataset_test,
        shuffle=False, sampler=sampler,
        batch_size=1,
    )

    if rank == 0:
        print(f"Unique testing samples: {len(dataset_test)}")

    # create model and load checkpoint
    args, model = create_model(args)
    checkpoint = torch.load("../output_dir/"+args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    model.eval()

    if rank == 0:
        print(f"Loaded pre-trained model from {args.resume}")

    #print("Class Distribution:")
    #analyze_class_distribution(dataset, device)

    if args.task == "classification":
        results = classification_validation(args, model, data_loader, device)
    elif args.task == "segmentation":
        results = segmentation_validation(args, model, data_loader, device)
    elif args.task == "change":
        results = change_validation(args, model, data_loader, device)

    cleanup()


def check_required_args(args):
    missing_args = []

    if args.checkpoint == "":
        missing_args.append("--checkpoint argument is required. [filename in ../output_dir/]")
    if args.dataset == "":
        missing_args.append("--dataset argument is required. [resisc45, ucmerced, ..., potsdam]")
    if args.gpu == "":
        missing_args.append("--gpu argument is required [0,1,3,4,5,6,7].")

    if missing_args:
        for error in missing_args:
            print(f"Error: {error}")
        sys.exit(1)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    check_required_args(args)
    args = config(args)
    args.resume = "" # remove pretrained model
    args.port = "13579"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"GPUs: {args.gpu}")

    args.world_size = torch.cuda.device_count()
    mp.spawn(main, nprocs=args.world_size, args=(args,), join=True)