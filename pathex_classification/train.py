import argparse
import datetime
import os
import time
import warnings

import albumentations as album
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
# import torchvision.transforms as T
from albumentations import DualTransform
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, roc_auc_score)
from timm import models
from timm.data import Mixup
# from timm.loss import SoftTargetCrossEntropy
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from dataset.torchdataset import (TestDataset, find_classes, make_dataset,
                                  train_test_split_class)
from utils import utils
from utils import utils_torch as UT
from utils.logger import Logger
from utils.some_loss import (LabelSmoothingCrossEntropy, LDAMLoss,
                             SoftmaxEQLLoss, SoftmaxEQV2LLoss,
                             SoftTargetCrossEntropy)

# from torchvision import models


warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser('PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
group.add_argument('--data-dir', type=str, default='/mnt/d/Data_set/CRC_HE/NCT-CRC-HE-100K',
                   help='Train data directory.')
group.add_argument('--train-ratio', type=float, default=0.8,
                   help='Ratio to split train and validation dataset.')
group.add_argument('--num-classes', type=int, default=9,
                   help='Number of label classes (default is None, will be same as the dataset.')
group.add_argument('--batch-size', type=int, default=2,
                   help='Input batch size for training (default: 128).')
group.add_argument('--valid-batch-size', type=int, default=2,
                   help='Input batch size for validation (default: 128).')
group.add_argument('--pin_memory', type=utils.str2bool, default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--num_workers', type=int, default=2,
                   help='Number of workers in DataLoader to load data more efficient.')
group.add_argument('--drop_last', type=utils.str2bool, default=False,
                   help='Drop last batch of the train dataloader, True when enable mixup (default: False).')
group.add_argument('--test-dir', type=str, default='',
                   help='Test data directory.')
# Log parameters
group = parser.add_argument_group('Log parameters')
group.add_argument('--save-dir', type=str, default='./checkpoints/',
                   help='Directory to save checkpoints and logs.')
group.add_argument('--log-interval',  default=50,
                   type=int, help='dataset number workers')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--image-size', type=int, default=224,
                   help='Image size, which is the input size of the model.')
group.add_argument('--pretrained', type=utils.str2bool, default=False,
                   help='Start with pretrained version of specified network (if avail).')
# group.add_argument('--resume', type=str, default=None,
#                     help='Resume model and optimizer state from checkpoint (default: None).')
group.add_argument('--start-epoch', type=int, default=None,
                   help='Start epoch will overwrite resume start epoch (default: None).')
group.add_argument('--resume', type=str, default='checkpoints/hornet_cls_20230217-225735/model_27_0.9669.pth.tar',
                   help='Resume model and optimizer state from checkpoint (default: None).')
# group.add_argument('--resume', type=str, default='checkpoints/hornet_cls_20230222-154047/model_99_0.9883.pth.tar',
#                     help='Resume model and optimizer state from checkpoint (default: None).')
group.add_argument('--dino_path', type=str, default=None,
                   help='Dino pretrained model checkpoint.')

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt-betas', type=float, default=(0.9, 0.99),
                   help='Optimizer Betas (default: None, use opt default).')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9).')
group.add_argument('--weight-decay', type=float, default=2e-5,
                   help='Weight decay (default: 2e-5).')
group.add_argument('--clip-mode', type=str, default='norm',
                   help='Gradient clipping mode. One of ("norm", "value")')
group.add_argument('--clip-grap', type=float, default=None,
                   help='Max norm of clipping gradient (default: None, no clipping).')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--epochs', type=int, default=100,
                   help='Number of epochs to train (default: 300).')
group.add_argument('--lr', type=float, default=1.25e-3,
                   help='Learning rate for training (default: 1.25e-3).')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                   help='Epochs to warmup LR, if scheduler supports.')
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                   help='Minimum learning rate for warmup (default: 1e-5)')

# criterion
group = parser.add_argument_group('Criterion parameters')
group.add_argument('--weight', default='None', type=str,
                   help='class weight')

# Augmentation parameters
group = parser.add_argument_group('Augmentation parameters')
group.add_argument('--mixup', type=float, default=0.2,
                   help='mixup alpha, mixup enabled if > 0. (default: 0.)')
group.add_argument('--cutmix', type=float, default=0.2,
                   help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
group.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                   help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
group.add_argument('--mixup_prob', type=float, default=1.0,
                   help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup_switch_prob', type=float, default=0.5,
                   help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup_mode', type=str, default='batch',
                   help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--mixup_off_epoch', default=60, type=int, metavar='N',
                   help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.,
                   help='Label smoothing (default: 0.1)')

# Distributed
group = parser.add_argument_group('Distributed parameters')
group.add_argument('--use_gpu', type=utils.str2bool, default=True,
                   help="Should we use GPU to training? (default: True).")
group.add_argument('--dist_backend', type=str, default='nccl',
                   help='Backend for distributed.')
group.add_argument('--dist_url', type=str, default='tcp://localhost:11223',
                   help='Url used to set up distributed training.')
group.add_argument('--rank', type=int, default=0,
                   help='Node rank for distributed training.')
group.add_argument('--local_rank', type=int, default=0,
                   help='Node rank for distributed training.')
group.add_argument('--world_size', type=int, default=1,
                   help='Number of nodes for distributed training.')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42,
                   help='random seed')
group.add_argument('--amp', type=utils.str2bool, default=False,
                   help='Automatic mixed precision training enable.')
group.add_argument('--c-verbose', type=int, default=2,
                   help='Logger console verbose option in (0 is debug, 1 is info, 2 is warning) (default is 2)')

args = parser.parse_args()

# disable other logger
Logger.disable_existing_loggers()


def main(args):
    UT.seed_everything(args.seed)
    # Setup ddp
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    if args.distributed:
        if 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.getenv('LOCAL_RANK'))
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

    print(
        f'distributed: {args.distributed}, rank: {args.rank}, world_size: {args.world_size}, local_rank: {args.local_rank}')

    args.log_path, args.checkpoint_path = utils.create_save_path(
        args.save_dir, False)
    args.logger_config = Logger(
        args.log_path+'/config.log', c_verbose=args.c_verbose, f_verbose=1)
    args.logger = Logger(args.log_path+'/train.log',
                         c_verbose=args.c_verbose, f_verbose=1)

    if args.weight.lower() != 'none':
        args.weight = [float(x) for x in args.weight.split(',')]
    else:
        args.weight = None

    if UT.is_main_process():
        args.logger_config.info(utils.format_arg(args))

    # Create transforms and data loader
    transform_train, transform_valid = get_albumentation_transform(args)

    root = os.path.abspath(args.data_dir)
    # classes is the name of the folder
    classes, class_to_idx = find_classes(root)
    samples = make_dataset(root, class_to_idx)
    samples = np.array(samples, dtype=object)
    if UT.is_main_process():
        args.logger_config.info(f'class to idx: {class_to_idx}')

    train_samples, vadlid_samples = train_test_split_class(
        samples, train_ratio=args.train_ratio, shuffle=True)

    train_dataset = TestDataset(train_samples, transform=transform_train)
    valid_dataset = TestDataset(vadlid_samples, transform=transform_valid)

    if args.test_dir:
        test_root = os.path.abspath(args.test_dir)
        test_classes, test_class_to_idx = find_classes(test_root)
        test_samples = make_dataset(test_root, test_class_to_idx)
        test_dataset = TestDataset(test_samples, transform=transform_valid)

    # Setup mixup / cutmix
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None

    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=len(classes))
        mixup_fn = Mixup(**mixup_args)
        args.drop_last = True
        if UT.is_main_process():
            args.logger_config.info(f"Mixup enable {mixup_active}")

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        if args.test_dir:
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = None
        if args.test_dir:
            test_sampler = None

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=args.pin_memory,
                              sampler=train_sampler, drop_last=args.drop_last, persistent_workers=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.valid_batch_size,
                              num_workers=args.num_workers, pin_memory=args.pin_memory,
                              sampler=valid_sampler, persistent_workers=True)
    if args.test_dir:
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.valid_batch_size,
                              num_workers=args.num_workers, pin_memory=args.pin_memory,
                              sampler=test_sampler, persistent_workers=True)
    if UT.is_main_process():
        args.logger.info(
            f"There are {len(train_loader.dataset)} for training, {len(valid_loader.dataset)} for validation")
        if args.test_dir:
            args.logger.info(
            f"There are {len(test_loader.dataset)} for test")

    # Create model
    model = models.resnetv2_50d(pretrained=args.pretrained, num_classes=len(classes))

    if UT.is_main_process():
        args.logger.info(
            f"Mode is {model.__class__.__name__}")

    if args.dino_path:
        data = torch.load(args.dino_path)
        teacher = data['teacher']
        teacher = {k.replace('module.backbone.', ''): v.cpu()
                   for k, v in teacher.items() if not 'head' in k}
        msg = model.load_state_dict(teacher, strict=False)
        print(f'load from dino {msg}')

    if args.use_gpu:
        model = model.to(f'cuda:{args.local_rank}')

    # Enable distributed training
    if args.distributed:
        model = DDP(model, device_ids=[
                    args.local_rank], find_unused_parameters=False)

    # Create loss function
    if args.weight:
        args.weight = torch.FloatTensor(args.weight)
        if args.use_gpu:
            args.weight = args.weight.to(f'cuda:{args.local_rank}')
    if mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        train_loss_fn = SoftTargetCrossEntropy(args.weight)
    else:
        train_loss_fn = nn.CrossEntropyLoss(
            args.weight, label_smoothing=args.smoothing)

    valid_loss_fn = nn.CrossEntropyLoss()
    if UT.is_main_process():
        args.logger_config.info(f'Loss function: {str(train_loss_fn)}')

    # TODO: Create different optimizer
    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters(),
                                 betas=args.opt_betas, weight_decay=args.weight_decay)
    if UT.is_main_process():
        args.logger_config.info(f'Optimizer: {str(optimizer)}')

    save_best = UT.SaveBestModel(epoch=None, epochs=args.epochs, model=model,
                                 checkpoint_path=args.checkpoint_path, train_acc=None,
                                 monitor_value=0., save_every=False, best=0.)
    fp16_scaler = None
    if args.amp:
        fp16_scaler = GradScaler()

    # Resume training
    resume_epoch = {'epoch': 0}
    if args.resume:
        UT.resume_checkpoint(args.resume, run_variables=resume_epoch, model=model,
                             optimizer=optimizer, fp16_scaler=fp16_scaler)
    start_epoch = resume_epoch['epoch']
    if args.start_epoch is not None:
        start_epoch = args.start_epoch

    # TODO: create slective scheduler
    optimizer_scheduler = UT.WarmupCosineLR(
        optimizer, args.warmup_lr, args.lr, args.warmup_epochs, args.epochs, start_epoch)
    if UT.is_main_process():
        args.logger_config.info(f'Learning decay: {str(optimizer_scheduler)}')

    start = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        metrics_train = train_one_epoch(
            epoch, model, train_loader, optimizer, train_loss_fn, args, fp16_scaler, mixup_fn=mixup_fn)
        metrics_valid = validate(
            epoch, model, valid_loader, valid_loss_fn, args)
        optimizer_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        metrics_train['lr'] = lr

        # process result and get metrics
        valid_df = pd.DataFrame(metrics_valid['result'])
        valid_report = get_metrics(valid_df)
        printed = f'Valid metrics: [{epoch:>03d}/{args.epochs}], AUC: {valid_report["roc_auc"]:.4f}, precision: {valid_report["precision"]:.4f}, recall: {valid_report["recall"]:.4f}, fscore: {valid_report["fscore"]:.4f}'
        metrics_valid.pop('result')
        metrics_valid.update(valid_report)

        # Sve metrics
        save_dict = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch+1,
        )
        if fp16_scaler:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()

        if UT.is_main_process():
            print(printed)
            args.logger.info(printed)
            utils.update_summary(epoch, metrics_train, metrics_valid,
                                 os.path.join(
                                     args.checkpoint_path, 'summary.csv'),
                                 write_header=epoch == start_epoch)

            # Save best model only
            save_best.epoch = epoch
            save_best.monitor_value = metrics_valid["acc1"]
            save_best.model = model
            state = save_best.run(save_dict)

            print(
                f'{epoch + 1} epoches total time: {str(datetime.timedelta(seconds=int(time.time() - start)))}')
        
        # Test
        if args.test_dir and epoch > 5 and (epoch + 1) % 1 == 0:
            metrics_test = validate(
                epoch, model, test_loader, valid_loss_fn, args, header='Test: ')

            # process result and get metrics
            test_df = pd.DataFrame(metrics_test['result'])
            try:
                test_report = get_metrics(test_df)
                test_printed = f'Test metrics: [{epoch:>03d}/{args.epochs}], AUC: {test_report["roc_auc"]:.4f}, precision: {test_report["precision"]:.4f}, recall: {test_report["recall"]:.4f}, fscore: {test_report["fscore"]:.4f}'
            except:
                test_printed = ''
                pass

            metrics_test.pop('result')
            metrics_test.update(test_report)
            if UT.is_main_process():
                print(test_printed)
                args.logger.info(test_printed)
                utils.update_summary(epoch, metrics_train, metrics_test,
                                 os.path.join(
                                     args.checkpoint_path, 'summary_test.csv'),
                                 write_header=epoch == start_epoch)


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args,
                    loss_scaler=None, model_ema=None, mixup_fn=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    model.train()
    metric_logger = UT.MetricLogger(logger=args.logger, delimiter='  ')
    header = 'Training: [{}/{}]'.format(epoch, args.epochs)
    for batch_idx, (input, target) in enumerate(metric_logger.log_every(loader, args.log_interval, header)):
        batch_size = input.size(0)
        last_batch = batch_size != args.batch_size
        if args.use_gpu:
            input = input.to(f'cuda:{args.local_rank}')
            target = target.to(f'cuda:{args.local_rank}')

        if mixup_fn is not None:
            input, target = mixup_fn(input, target)

        if args.amp:
            with autocast():
                output = model(input)
                loss = loss_fn(output, target)
        else:
            output = model(input)
            loss = loss_fn(output, target)

        optimizer.zero_grad()
        # print(output.shape, target.shape)

        # acc1, acc5 = UT.accuracy(output, target, topk=(1, 5), ratio=1.0)

        if args.amp and loss_scaler is not None:
            loss_scaler.scale(loss).backward()
            if args.clip_grap is not None:
                UT.choose_clip_grad(model.parameters(),
                                    value=args.clip_grap, mode=args.clip_mode)
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            loss.backward()
            if args.clip_grap is not None:
                UT.choose_clip_grad(model.parameters(),
                                    value=args.clip_grap, mode=args.clip_mode)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        if args.distributed:
            reduced_loss = UT.reduce_tensor(loss.data, args.world_size)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()
        metric_logger.update(loss=reduced_loss.item())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(epoch, model, loader, loss_fn, args, header='Validate: '):
    model.eval()
    metric_logger = UT.MetricLogger(logger=args.logger, delimiter='  ')
    header = header + '[{}/{}]'.format(epoch, args.epochs)
    result = dict(prob=[], pred_label=[], label=[])
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(metric_logger.log_every(loader, args.log_interval, header)):
            batch_size = input.size(0)
            last_batch = batch_size != args.batch_size

            if args.use_gpu:
                input = input.to(f'cuda:{args.local_rank}')
                target = target.to(f'cuda:{args.local_rank}')
            # forward
            output = model(input)
            loss = loss_fn(output, target)

            pred = torch.softmax(output, dim=1)
            prob, pred_label = torch.max(pred, dim=1)

            acc1 = UT.accuracy(output, target, topk=(1,), ratio=1.0)
            if isinstance(acc1, (tuple, list)):
                acc1 = acc1[0]

            # map reduce
            if args.distributed:
                reduced_loss = UT.reduce_tensor(loss.data, args.world_size)
                acc1 = UT.reduce_tensor(acc1, args.world_size)
                prob = UT.sync_tensor_across_gpus(prob)
                pred_label = UT.sync_tensor_across_gpus(pred_label)
                target = UT.sync_tensor_across_gpus(target)
            else:
                reduced_loss = loss.data

            # update metrics
            torch.cuda.synchronize()
            metric_logger.update(loss=reduced_loss.item())
            metric_logger.update(acc1=acc1.item())

            result['prob'].extend(prob.data.tolist())
            result['pred_label'].extend(pred_label.data.tolist())
            result['label'].extend(target.data.tolist())

    metrics = {k: meter.global_avg for k,
               meter in metric_logger.meters.items()}
    metrics['result'] = result
    return metrics


def get_torchvision_transform(args):
    import torchvision.transforms as T
    flip_and_color_jitter = T.Compose([
        T.RandomApply([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(degrees=45, translate=(0.2, 0.2))
        ]),
        T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4,
                          saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.2),
        ]),
    ])
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_transforms = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        flip_and_color_jitter,
        # autoaugment.AutoAugment(),
        # autoaugment.RandAugment(),
        utils.GaussianBlur(0.2),
        # utils.CutOut(0.2, value='random'),
        normalize
    ])

    valid_transforms = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        normalize
    ])

    return train_transforms, valid_transforms


def get_metrics(df):
    df.loc[df.pred_label == 0, 'prob'] = 1 - df.prob
    y_true = df['label']
    y_pred = df['pred_label']
    y_score = df['prob']

    report = dict()
    report['precision'], report['recall'], report['fscore'], support = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1)
    report['roc_auc'] = roc_auc_score(y_true, y_score)
    report['acc_balanced'] = balanced_accuracy_score(y_true, y_pred)
    return report


def add_salt_pepper(img, ps=0.01, pp=0.01):
    """
    add salt pepper noise to image
    param: img: input image, uint8 [0, 255]
    param: ps:  probability of salt noise, which is white noise, default is 0.01
    param: pp:  probability of peper noise, which is black noise, default is 0.01
    return image with salt pepper noise, [0, 255]
    """
    h, w = img.shape[:2]
    mask = np.random.choice((0, 0.5, 1), size=(h, w), p=[pp, (1-ps-pp), ps])

    img_out = img.copy()
    img_out[mask == 1] = 255
    img_out[mask == 0] = 0
    return img_out


class RandomSaltPepperNoise(DualTransform):
    def __init__(self, prob, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.prob = prob

    def apply(self, img, **params):
        prob = np.random.uniform(0, self.prob, (1,)).item()
        pp = np.random.uniform(0, prob, (1,)).item()
        ps = prob - pp
        return add_salt_pepper(img, ps=ps, pp=pp)


def get_albumentation_transform(args):
    normal_transform = album.Sequential([
        album.OneOf([album.VerticalFlip(),
                     album.HorizontalFlip()]),
        album.OneOf([
            album.GaussNoise(var_limit=(10., 50.0), mean=0),
            album.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5))
        ]),
        album.RandomGamma(gamma_limit=(20, 130))
    ])
    strong_transform = album.SomeOf([
        album.OneOf([
            album.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1, p=0.7),
            album.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True),
            album.ToGray()
        ]),
        album.OneOf([
            album.GaussianBlur(blur_limit=(3, 9), sigma_limit=10),
            album.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5)),
            album.MotionBlur(blur_limit=21)
        ]),
        album.Affine(
            scale=(0.95, 1.2), translate_percent=0.1, rotate=10, shear=10, p=0.3),
        RandomSaltPepperNoise(0.1),
        album.CoarseDropout(
            max_holes=2,
            max_height=50,
            max_width=50,
            fill_value=0,
        ),
        album.RandomGridShuffle(grid=(2, 2)),
        album.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            border_mode=4,
        ),
        album.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
        album.ImageCompression(
            quality_lower=90,
            quality_upper=100,
        )
    ],
        n=2,
        replace=False)

    train_transform = album.Compose([
        # album.Resize(args.image_size, args.image_size),
        album.RandomResizedCrop(args.image_size, args.image_size, 
                                scale=(0.4, 1.), ratio=(0.75, 1.35),
                                interpolation=1, p=0.5),
        normal_transform,
        strong_transform,
        album.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    valid_transform = album.Compose([
        album.Resize(args.image_size, args.image_size),
        album.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, valid_transform


if __name__ == '__main__':
    mp.set_sharing_strategy('file_system')
    mp.set_start_method('spawn')
    start_time = time.time()
    main(args)
    if UT.is_main_process():
        print(
            f'All files were created, time took: {str(datetime.timedelta(seconds=int(time.time() - start_time)))} seconds')
