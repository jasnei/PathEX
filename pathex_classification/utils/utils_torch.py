import datetime
import itertools
import os
import random
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import sampler


class DistributedTwoSteamBatchSampler(sampler.Sampler):
    def __init__(self, unlabeled_indices, labeled_indices, unlabeled_bs, 
                 labeled_bs, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.unlabeled_indices = unlabeled_indices
        self.labeled_indices = labeled_indices
        self.unlabeled_bs = unlabeled_bs
        self.labeled_bs = labeled_bs
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.unlabeled_num_samples = int(np.ceil(len(unlabeled_indices) / self.num_replicas))
        self.labeled_num_samples = int(np.ceil(len(labeled_indices) / self.num_replicas))
        self.unlabeled_total_size = self.unlabeled_num_samples * self.num_replicas
        self.labeled_total_size = self.labeled_num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            unlabeled_indices = torch.randperm(len(self.unlabeled_indices), generator=g).tolist()
            labeled_indices = torch.randperm(len(self.labeled_indices), generator=g).tolist()
        else:
            unlabeled_indices = list(range(len(self.unlabeled_indices)))
            labeled_indices = list(range(len(self.labeled_indices)))

        # add extra sample to make it evenly divisible
        unlabeled_indices += unlabeled_indices[:(self.unlabeled_total_size - len(unlabeled_indices))]
        labeled_indices += labeled_indices[:(self.labeled_total_size - len(labeled_indices))]

        # subsample
        unlabeled_indices = unlabeled_indices[self.rank:self.unlabeled_total_size:self.num_replicas]
        labeled_indices = labeled_indices[self.rank:self.labeled_total_size:self.num_replicas]
        
        unlabeled_indices_rank = self.unlabeled_indices[unlabeled_indices]
        labeled_indices_rank = self.labeled_indices[labeled_indices]
        unlabeled_iter = iterate_once(unlabeled_indices_rank, False)
        labeled_iter = iterate_eternally(labeled_indices_rank, False)
        for unlabeled_batch, labeled_batch in zip(
                grouper(unlabeled_iter, self.unlabeled_bs),
                grouper(labeled_iter, self.labeled_bs)):
            unlabeled_batch = list(unlabeled_batch)
            labeled_batch = list(labeled_batch)
            yield torch.cat((labeled_indices_rank[labeled_batch],
                              unlabeled_indices_rank[unlabeled_batch]))

    def __len__(self):
        return self.unlabeled_num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch


class TwoStreamBatchSampler(sampler.Sampler):
    """Iterate two sets of indices
    """

    def __init__(self,
                 primary_indices,
                 secondary_indices,
                 batch_size,
                 secondary_batch_size,
                 shuffle=True):
        """
            primary_indices: unlabel indices
            secondary_indices: label indices
            batch_size: batch size for unlabeled
            secondary_batch_size: batch size for labeled
        """
        self.primary_indices = primary_indices
        self.primary_batch_size = batch_size  # - secondary_batch_size
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.shuffle = shuffle
        self.num_samples = int(
            len(self.primary_indices) / self.primary_batch_size)
        # assert len(self.primary_indices) >= self.primary_batch_size > 0
        # assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices, self.shuffle)
        secondary_iter = iterate_eternally(self.secondary_indices, self.shuffle)
        for primary_batch, secondary_batch in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size)):
            primary_batch = list(primary_batch)
            secondary_batch = list(secondary_batch)
            # print(self.secondary_indices[secondary_batch])
            yield torch.cat((self.secondary_indices[secondary_batch],
                              self.primary_indices[primary_batch]))
        # for primary_batch in grouper(primary_iter, self.primary_batch_size):
        #     print(self.primary_indices[list(primary_batch)])
        # return (self.secondary_indices[secondary_batch] + self.primary_indices[primary_batch]
        #         for (primary_batch, secondary_batch) in zip(
        #             grouper(primary_iter, self.primary_batch_size),
        #             grouper(secondary_iter, self.secondary_batch_size)))

    def __len__(self):
        return self.num_samples


def iterate_once(iterable, shuffle=True):
    if shuffle:
        return torch.randperm(len(iterable)).tolist()
    else:
        return list(range(len(iterable)))


def iterate_eternally(indices, shuffle=True):

    def infinite_shuffles(shuffle=True):
        while True:
            if shuffle:
                yield torch.randperm(
                    len(indices)).tolist()  #np.random.permutation(indices)
            else:
                yield list(range(len(indices)))  # indices

    return itertools.chain.from_iterable(infinite_shuffles(shuffle))


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)


def get_separate_indices(targets, n_samples: int = 5, shuffle: bool = True):
    """separate label and unlabel indices"""
    label_indices = []
    unlabel_indices = []
    unique = torch.unique(targets)
    for uniq in unique:
        target_indices = torch.nonzero(targets == uniq)
        if shuffle:
            temp = torch.randperm(target_indices.shape[0])
            second_indices = temp[:n_samples]
            second_indices_left = temp[n_samples:]
            selected_indices = target_indices[second_indices]
            unselected_indices = target_indices[second_indices_left]
        else:
            selected_indices = target_indices[:n_samples]
            unselected_indices = target_indices[n_samples:]

        label_indices.append(selected_indices)
        unlabel_indices.append(unselected_indices)

    return torch.cat(label_indices).squeeze(), torch.cat(unlabel_indices).squeeze()
    

def accuracy(output, target, topk=(1,), ratio=100.):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if output.dim() != target.dim():
        target = target.reshape(1, -1).expand_as(pred)
    correct = pred.eq(target)
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * ratio / batch_size for k in topk]


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
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
    def __init__(self, logger=None, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.msg = ''
        self.logger = logger

    def update(self, **kwargs):
        for k, v in kwargs.items():
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
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters} '
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters} ',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.msg = log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB)
                else:
                    self.msg = log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time))
                if is_main_process():
                    print(self.msg)
                    if self.logger:
                        self.logger.info(self.msg)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.msg = '{} Total time: {} ({:.4f} s/it)'.format(
            header, total_time_str, total_time / len(iterable))
        if is_main_process():
            print(self.msg)
            if self.logger:
                self.logger.info(self.msg)


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def sync_tensor_across_gpus(t):
    if t is None:
        return None
    group = dist.group.WORLD
    group_size = dist.get_world_size(group)
    gather_t_tensor = [torch.zeros_like(t) for _ in range(group_size)]
    dist.all_gather(gather_t_tensor, tensor=t)
    return torch.cat(gather_t_tensor, dim=0)


def sync_object_across_gpus(obj):
    """sync object across process, iterative item, such as tuple, list. will return as list"""
    if obj is None:
        return None
    group = dist.group.WORLD
    group_size = dist.get_world_size(group)
    object_list = [obj for _ in range(group_size)]
    dist.all_gather_object(object_list, obj)
    return [item for sublist in object_list for item in sublist]


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


def seed_everything(seed: int=42) -> None:
    """set up all random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


class SaveBestModel:
    """
    Description:
        - Save best model & last model only, for the number is increasing(such as acc), but not for loss at the moment
    Parameters:
        - epoch: input, the epoch your model training
        - epoch: input, total epochs of the whole training process
        - monitor_value: input, the value you want to monitor, here always monitor valid_acc, if you want to monitor other value,
                         you might need to change the run function condition
        - best: input, best monitor value
        - checkpoint_papth, input, where to save the checkpoint
        - model: input the model you want to save
    """

    def __init__(self, epoch: int, epochs: int, model, checkpoint_path: str, 
                 monitor_value: float = 0.0, monitor_mode: str='greater', 
                 train_acc: float = None, best: float = 0.0, save_every: bool = False,) -> None:
        self.epoch = epoch
        self.epochs = epochs
        self.model = model
        self.checkpoin_path = checkpoint_path
        self.monitor_mode = monitor_mode.lower()
        self.monitor_value = monitor_value
        assert self.monitor_mode in ('greater', 'less'), f'monitor_mode must be in ("greater", "less") but got {self.monitor_mode}'
        self.train_acc = train_acc
        self.save_every = save_every

        # Set up best value for start epoch not 0
        if epoch is not None:
            if self.monitor_mode == 'less':
                if best < 100:
                    self.best = 100.
                else:
                    self.best = best
            elif self.monitor_mode == 'greater':
                if best >= 0.5:
                    self.best = 0.5
                else:
                    self.best = best
        else:
            self.best = best

    def run(self, state_dict=None):
        if state_dict is None:
            # if DP or normal
            if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.distributed.DistributedDataParallel):
                state_dict = {
                    'state_dict': self.model.module.state_dict()
                }
            else:
                state_dict = {
                    'state_dict': self.model.state_dict()
                }

        if self.train_acc is not None:
            save_dir = os.path.join(
                self.checkpoin_path, f'model_{self.epoch}_{self.train_acc:.4f}_{self.monitor_value:.4f}.pth.tar')
            msg = f'saved epoch {self.epoch}_{self.train_acc:.4f}_{self.monitor_value:.4f} model'
        else:
            save_dir = os.path.join(
                self.checkpoin_path, f'model_{self.epoch}_{self.monitor_value:.4f}.pth.tar')
            msg = f'saved epoch_{self.epoch}_{self.monitor_value:.4f} model'

        if self.save_every:
            torch.save(state_dict, save_dir)
            print(msg)
            return 'every'
        else:
            # Save best model only
            if self.epoch == 0:
                # Save every first epoch
                self.best = self.monitor_value
                torch.save(state_dict, save_dir)
                print(msg)
                return 'first'
            elif (self.epoch + 1) == self.epochs:
                # Last epoch
                save_dir = os.path.join(
                    self.checkpoin_path, f'last_{self.monitor_value:.4f}.pth.tar')
                torch.save(state_dict, save_dir)
                print(f'saved model last, {save_dir}')
                return 'last'
            else:
                if self.monitor_mode == 'greater':
                    if self.best < self.monitor_value:
                        self.best = self.monitor_value
                        torch.save(state_dict, save_dir)
                        print(msg)
                        return 'best'
                elif self.monitor_mode == 'less':
                    if self.best > self.monitor_value:
                        self.best = self.monitor_value
                        torch.save(state_dict, save_dir)
                        print(msg)
                        return 'best'


def weights_init_name_parameters(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            if param.data.dim() != 1:
                param.data = nn.init.kaiming_normal_(param.data)
            else:
                param.data = nn.init.normal_(param.data, 0)
        if "bias" in name:
            param.data = nn.init.constant_(param.data, 10)


def weights_init_modules(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight = nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias = nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight = nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias = nn.init.constant_(m.bias, 0)


def torch_loader(file):
    return torch.load(file)


def test_time_augmentation(image, model, device, transform, transform_times):
    tensor_images = torch.unsqueeze(transform(image=image)["image"], 0)
    if transform_times > 1:
        for i in range(1, transform_times):
            tensor_image_0 = torch.unsqueeze(transform(image=image)["image"], 0)
            tensor_images = torch.cat((tensor_image_0, tensor_images), 0)
            
    # print(transformed_images.shape)
    ######## Show images ##############
    # fig = plt.figure(figsize=(25, 5))
    # for i in range(transform_times):
    #     ax = fig.add_subplot(1, 5, i+1)
    #     ax.imshow(transformed_images[i])
    #     #ax.set_title(show_list[i])
    # plt.tight_layout()
    # plt.show()

    ########## Inference #############
    imgs = tensor_images.to(device)
    preds = model(imgs)
    _, pred_labels = torch.max(preds, 1)
    label = map_tta_result(pred_labels.cpu().numpy())
    # print(pred_labels.cpu().numpy())
    # print(label)
    return label


def map_tta_result(pred_labels):

    count  = np.bincount(pred_labels)
    label  = np.argmax(count)

    return label


# def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
#     warmup_schedule = np.array([])
#     warmup_iters = warmup_epochs * niter_per_ep
#     if warmup_epochs > 0:
#         warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

#     iters = np.arange(epochs * niter_per_ep - warmup_iters)
#     schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

#     schedule = np.concatenate((warmup_schedule, schedule))
#     assert len(schedule) == epochs * niter_per_ep
#     return schedule


def cosine_scheduler(lr_base, lr_final, epochs, niter_per_ep, warmup_epochs=0, lr_start=1e-5):
    warmup_schedule = np.array([])

    # Fix bug, if warmup epochs greater than epochs
    # set warmup_epochs = 0, or could be set to epochs
    if warmup_epochs > epochs:
        warmup_epochs = 0
    warmup_iters = warmup_epochs * niter_per_ep

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(lr_start, lr_base, warmup_iters)
    
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = lr_final + 0.5 * (lr_base - lr_final) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_start, lr_base, warm_up=0, T_max=10, cur=0):
        """
        Description:
            - get warmup consine lr scheduler
        
        Arguments:
            - optimizer: (torch.optim.*), torch optimizer
            - lr_start: (float), minimum learning rate
            - lr_base: (float), maximum learning rate
            - warm_up: (int),  warm_up epoch or iteration
            - T_max: (int), maximum epoch or iteration
        
        Example:
            <<< epochs = 100
            <<< warm_up = 5
            <<< cosine_lr = WarmupCosineLR(optimizer, 1e-9, 1e-3, warm_up, epochs)
            <<< lrs = []
            <<< for epoch in range(epochs):
            <<<     optimizer.step()
            <<<     lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
            <<<     cosine_lr.step()
            <<< plt.plot(lrs, color='r')
            <<< plt.show()
        
        """
        self.lr_start = lr_start
        self.lr_base = lr_base
        self.warm_up = warm_up
        self.T_max = T_max
        self.cur = cur    # current epoch or iteration
        self.lrs = cosine_scheduler(lr_base, lr_start, T_max, 1, warm_up, lr_start)
        self.lrs = self.lrs.tolist()
        self.lrs.append(lr_start)
        
        super().__init__(optimizer, -1)

    def get_lr(self):
        lr = self.lrs[self.cur]
        self.cur += 1
        return [lr for base_lr in self.base_lrs]

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'optimizer={self.optimizer.__class__.__name__}, '
        format_string += f'lr_start={self.lr_start}, '
        format_string += f'lr_base={self.lr_base}, '
        format_string += f'warm_up={self.warm_up}, '
        format_string += f'T_max={self.T_max}, '
        format_string += f'cur={self.cur-1}'
        format_string += ')'
        return format_string


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def resume_checkpoint(checkpoint_file, run_variables=None, **kwargs):
    if not os.path.isfile(checkpoint_file):
        return
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                checkpoint_ = checkpoint[key]
                if key == "model":
                    # model is ddp then checkpoint is ddp
                    if isinstance(value, torch.nn.DataParallel) or isinstance(value, torch.nn.parallel.distributed.DistributedDataParallel):
                        keys = list(checkpoint_.keys())[:2]
                        checkpoint_ddp = False
                        for k in keys:
                            if k.startswith('module'):
                                checkpoint_ddp=True
                                break
                        if checkpoint_ddp:
                            pass
                        else:
                            # checkpoint not ddp
                            checkpoint_ = {'module.' + k:v for k, v in checkpoint_.items()}
                    else:
                        checkpoint_ = {k.replace("module.", ""): v for k, v in checkpoint_.items()}
    
                msg = value.load_state_dict(checkpoint_, strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, checkpoint_file, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, checkpoint_file))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, checkpoint_file))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, checkpoint_file))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def get_model_name(model):
    return model.__class__.__name__


def load_vit_pretrained_weights(model, checkpoint_path, checkpoint_key):
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith('backbone.')}
        msg = model.load_state_dict(state_dict, strict=False)
        model_name = get_model_name(model)
        print(f'Load {model_name} pretrained done, state: {msg}')
    else:
        print('model is in random weights!!!')


def choose_clip_grad(parameters, value: float, mode: str = 'norm', norm_type: float = 2.0):
    """ Choose gradient clipping method

    Args:
        parameters (Iterable): model parameters to clip
        value (float): clipping value/factor/norm, mode dependant
        mode (str): clipping mode, one of 'norm', 'value', 'agc'
        norm_type (float): p-norm, default 2.0
    """
    if mode == 'norm':
        torch.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)
    elif mode == 'value':
        torch.nn.utils.clip_grad_value_(parameters, value)
    else:
        assert False, f"Unknown clip mode ({mode})."