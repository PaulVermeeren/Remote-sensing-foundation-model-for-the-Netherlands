import sysimport os#os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5" os.environ["PYTHONWARNINGS"] = "ignore:Failed to load image Python extension:"os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'os.environ['NCCL_TIMEOUT'] = '3600'os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'import datetimeimport jsonfrom pathlib import Pathimport numpy as npimport timeimport warningsimport signalimport torchimport torch.backends.cudnn as cudnnfrom torch.utils.tensorboard import SummaryWriterfrom torch.nn.parallel import DistributedDataParallel as DDPfrom torch.utils.data import DataLoader, DistributedSamplerimport torch.multiprocessing as mpimport timm.optim.optim_factory as optim_factoryparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))sys.path.append(parent_dir)import util.misc as miscfrom util.misc import NativeScalerWithGradNormCount as NativeScalerfrom util.misc import setup, get_open_port, cleanup, signal_handler, setup_printing, generate_unique_id, EarlyStoppingfrom misc import get_args_parser, create_modelfrom config import configfrom loader import CustomDataLoaderfrom engine import train_one_epochfrom validate import ValidationHandlerwarnings.filterwarnings("ignore")signal.signal(signal.SIGINT, signal_handler)def collate_fn(batch):    imgs = []    targets = []    for img, target in batch:        imgs.append(img)        targets.append(target)    return torch.stack(imgs, 0), targets  # imgs are stacked, but boxes and classes are in a listdef main(rank, args):    setup(args, rank, args.world_size)        # Fixed random seeds    seed = args.seed    torch.manual_seed(seed)    np.random.seed(seed)        # setup printing for master only    setup_printing(rank==0, args.disable_logging)     # Set up training equipment    torch.cuda.set_device(rank)    device = torch.device(args.device)    cudnn.benchmark = True          # load config for specific tasks    args = config(args)        if args.dataset == "dior":        custom_collate_fn = collate_fn    else:        custom_collate_fn = None        # dataset    dataset_train = CustomDataLoader(args, path=args.train_path, num_augmentations=args.num_augmentations)    dataset_val = CustomDataLoader(args, path=args.val_path, num_augmentations=0)    args.training_samples = int(len(dataset_train) / (args.num_augmentations+1)) * args.temporal_size    args.validation_samples = len(dataset_val) * args.temporal_size    print(f"Total training samples: {args.training_samples * (args.num_augmentations+1)}")    print(f"Unique training samples: {args.training_samples}")    print(f"Unique validation samples: {args.validation_samples}")        sampler_train = DistributedSampler(dataset_train, num_replicas=args.world_size, rank=rank, shuffle=True)        data_loader_train = DataLoader(        dataset_train,        shuffle=False, sampler=sampler_train,        batch_size=args.batch_size,        num_workers=2,        pin_memory=False,        drop_last=True,        persistent_workers=True,        prefetch_factor=2,        collate_fn=custom_collate_fn    )            sampler_train_val = DistributedSampler(dataset_val, num_replicas=args.world_size, rank=rank, shuffle=True)    data_loader_val = DataLoader(        dataset_val,        shuffle=False, sampler=sampler_train_val,        batch_size=args.batch_size,        num_workers=2,        pin_memory=False,        drop_last=True,        persistent_workers=False,        collate_fn=custom_collate_fn    )            # define model    args, model = create_model(args)    model_without_ddp = model    model.to(device)    model = DDP(model, device_ids=[rank], find_unused_parameters=False)                    args.total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)    print(f"Total trainable parameters {args.total_parameters}")        # only base_lr is specified    eff_batch_size = args.batch_size * args.accum_iter * args.world_size * args.temporal_size    if args.lr is None:  # only base_lr is specified        args.lr = args.blr * eff_batch_size / args.img_size    args.min_lr = args.lr * 0.01    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))    loss_scaler = NativeScaler()        if args.resume_head:        args.resume = args.resume_head # voor load_model methode        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)        print(f"Loaded pre-trained model from {args.resume}")            # logger    if rank == 0 and args.log_dir is not None and not args.disable_logging:        os.makedirs(args.log_dir, exist_ok=True)        args.tensorboard_log_dir = f"{args.log_dir}/{args.dataset}_{datetime.datetime.now().strftime('%Y-%m-%d')}_{args.tag}_{args.rand_id}/"        log_writer = SummaryWriter(log_dir=args.tensorboard_log_dir)                   # add args        args_dict = vars(args)        args_str = "\n\n".join([f"{k}: {v}" for k, v in args_dict.items()])        log_writer.add_text('arguments', args_str, global_step=0)                log_writer.flush()             else:        log_writer = None        print(f"Start training for {args.epochs} epochs")    start_time = time.time()            validation_handler = ValidationHandler(args, model, model_without_ddp, data_loader_val, optimizer, loss_scaler, log_writer, rank)           epochs_trained = 0     for epoch in range(args.start_epoch, args.epochs):        epochs_trained = epoch            data_loader_train.sampler.set_epoch(epoch)                train_stats = train_one_epoch(            model, data_loader_train,            optimizer, device, epoch, rank, loss_scaler,            log_writer=log_writer,            args=args,        )                if rank == 0 and log_writer is not None:            log_writer.add_scalar('average_train/loss', train_stats['loss'], epoch)                        # save every epoch            misc.save_model(                args=args, model=model, model_without_ddp=model_without_ddp,                 optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch + 1,                 rand_id=args.rand_id)            print(f"Saving at {epoch + 1} with id {args.rand_id}")                # intermediate saving and last epoch saving        if ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):            validation_handler.update(train_stats['loss'], epoch, device)                    # Synchronize all ranks after validation        torch.distributed.barrier()                            if not args.disable_logging:            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},                     'epoch': epoch, }                      if args.output_dir and rank == 0 and not args.disable_logging:            if log_writer is not None:                log_writer.flush()            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:                f.write(json.dumps(log_stats) + "\n")        # check for early stopping            if validation_handler.check_early_stop(device):            torch.distributed.barrier()            break                if rank == 0 and not args.disable_logging:        total_time = time.time() - start_time        total_time_str = str(datetime.timedelta(seconds=int(total_time)))        print('Training time {}'.format(total_time_str))                final_stats = {            'Epochs Trained': epochs_trained,            'Final Training Loss': train_stats['loss'],            'Total Training Time': total_time_str,        }                  args_str = "\n\n".join([f"{k}: {v}" for k, v in final_stats.items()])        log_writer.add_text('Final Statistics', args_str, global_step=1)                log_writer.close()            cleanup()    def check_required_args(args):    missing_args = []        if args.tag == "":        missing_args.append("--tag argument is required.")    if args.dataset == "":        missing_args.append("--dataset argument is required. [resisc45, ucmerced, ..., potsdam]")    if args.gpu == "":        missing_args.append("--gpu argument is required [0,1,3,4,5,6,7].")        if missing_args:        for error in missing_args:            print(f"Error: {error}")        sys.exit(1)    if __name__ == '__main__':    args = get_args_parser().parse_args()    args.rand_id = generate_unique_id()    if args.output_dir:        Path(args.output_dir).mkdir(parents=True, exist_ok=True)        check_required_args(args)            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu    args.world_size = torch.cuda.device_count()    args.port = get_open_port()    print(f"GPUs: {args.gpu}")    print(f"Port: {args.port}")    print(f"ID: {args.rand_id}")        mp.spawn(main, nprocs=args.world_size, args=(args,), join=True)