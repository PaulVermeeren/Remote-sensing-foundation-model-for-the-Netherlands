from torch.utils.data import Dataset

from loaders.loader_resisc_merced import CustomDataLoader as loader_resisc_merced
from loaders.loader_levircd import CustomDataLoader as loader_levircd
from loaders.loader_potsdam import CustomDataLoader as loader_potsdam
from loaders.loader_fair1m import CustomDataLoader as loader_fair1m
from loaders.loader_dior import CustomDataLoader as loader_dior
from loaders.loader_vegetatie_temp_highres import CustomDataLoader as loader_vegetatie_temp_highres
from loaders.loader_vegetatie_temp import CustomDataLoader as loader_vegetatie_temp
from loaders.loader_vegetatie_temp_ahn import CustomDataLoader as loader_vegetatie_temp_ahn
from loaders.loader_vegetatie_temp_single import CustomDataLoader as loader_vegetatie_temp_single
from loaders.loader_ahn import CustomDataLoader as loader_ahn

def CustomDataLoader(args, path, num_augmentations=0, prepend=""):
    if args.dataset == "ucmerced":
        dataloader = loader_resisc_merced(
          path,
          num_augmentations=num_augmentations,
          interpolate=256,
          maxdata=args.maxdata)
    elif args.dataset == "resisc45":
        dataloader = loader_resisc_merced(
          path,
          num_augmentations=num_augmentations,
          interpolate=args.interpolate,
          maxdata=args.maxdata)
    elif args.dataset == "levircd":
        dataloader = loader_levircd(
          path,
          num_augmentations=num_augmentations,
          temporal_size=args.temporal_size,
          in_chans=args.in_chans)
    elif args.dataset == "potsdam":
        dataloader = loader_potsdam(
          path,
          num_augmentations=num_augmentations,
          temporal_size=args.temporal_size,
          in_chans=args.in_chans)
    elif args.dataset == "vegetatie_temp":
        dataloader = loader_vegetatie_temp(
          path,
          num_augmentations=num_augmentations,
          maxdata=args.maxdata, prepend=prepend)
    else:
        print("Loader undefined")

    return dataloader
