tasks_dict = {
    "vegetatie_temp": "segmentation",
    "potsdam": "segmentation",
    "levircd": "change",
    "ucmerced": "classification",
    "resisc45": "classification",
}

def config(args):
    if tasks_dict[args.dataset]:
        args.task = tasks_dict[args.dataset]
    else:
        print("config.py -- dataset type task not configured")
        sys.exit(1)

    args.train_path = f'../../data/{args.dataset}/train/'
    args.val_path = f'../../data/{args.dataset}/val/'
    args.test_path = f'../../data/{args.dataset}/test/'

    if args.task == "segmentation":


        if args.dataset == "vegetatie_temp":
            args.batch_size = 1
            args.maxdata = 2150

            args.num_classes = 6

            args.img_size = 512
            args.temporal_size = 6
            args.in_chans = 6

            args.train_path = '../../data/vegetatie/oldsplit_4326linkerkant_vegetatie_train_split_random.json'
            args.val_path = '../../data/vegetatie/oldsplit_4326linkerkant_vegetatie_val_split_random.json'
            args.test_path = '../../data/vegetatie/oldsplit_4326linkerkant_vegetatie_test_split_random.json'

        elif args.dataset == "potsdam":
            args.batch_size = 16
            args.num_classes = 6

            args.img_size = 512
            args.temporal_size = 1
            args.in_chans = 4

            args.test_path = args.val_path


    elif args.task == "classification":
        if args.dataset == "ucmerced":
            args.batch_size = 30

            args.img_size = 256
            args.interpolate = 256
            args.num_classes = 21

            args.temporal_size = 1
            args.in_chans = 3

        elif args.dataset == "resisc45":
            args.batch_size = 30

            args.img_size = 256
            args.num_classes = 45

            args.temporal_size = 1
            args.in_chans = 3

    elif args.task == "change":
        if args.dataset == "levircd":
            args.batch_size = 1

            args.img_size = 1024
            args.num_classes = 2

            args.temporal_size = 2
            args.in_chans = 3

    return args