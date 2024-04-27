import torch
from tqdm import tqdm


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # 在数据处理里已经做了标准化了，所以这里不再需要做了
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)

        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def train_valid(model, loss_fn, optimizer, epoch, data_loader, train_valid=None, logger=None, print_freq=100):

    total_loader = len(data_loader)
    total = 0
    epoch_correct = 0
    epoch_acc = 0.
    epoch_loss = 0.

    if train_valid == "train":
        model.train()
        with tqdm(total=total_loader) as pbar:
            for i, (images, labels) in enumerate(data_loader):
                #=========Forward============
                batch_size = images.size(0)
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                images = images.squeeze(0)
                labels = labels.squeeze(0)

                pred = model(images)
                pred = pred.squeeze(1)

                #===========Loss=============
                loss = loss_fn(pred, labels)

                #==========Backward==========
                # Reset gradients, for the next accumulated batches
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, pred_label = torch.max(pred, dim=1)

                epoch_correct += (pred_label==labels).squeeze().sum().item()
                epoch_loss += loss.item()
                total += batch_size
                epoch_acc =  epoch_correct / total
                # print(f'batch_size: {batch_size}, total: {total}, epoch_correct: {epoch_correct}, epoch_acc: {epoch_acc}')

                if (i + 1) % print_freq == 0:
                    printed = f'<<< Training: Epoch: [{epoch:03d}], Step: [{i+1:03d}/{total_loader}], loss: {loss.item():>6.4f}, acc: {epoch_acc:.4f}'
                    # print(printed)
                    logger.info(printed)
                pbar.set_description(f'Training: Epoch: [{epoch:03d}], loss: {loss.item():.4f}, acc: {epoch_acc:.4f}')
                pbar.update(1)
                    

    elif train_valid == "valid":
        model.eval()
        with torch.no_grad():
            with tqdm(total=total_loader) as pbar:
                for i, (images, labels) in enumerate(data_loader):
                    batch_size = images.size(0)
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                    images = images.squeeze(0)
                    labels = labels.squeeze(0)

                    pred = model(images)
                    pred = pred.squeeze(1)

                    #===========Loss=============
                    loss = loss_fn(pred, labels)

                    prob, pred_label = torch.max(pred, dim=1)

                    epoch_correct += (pred_label==labels).squeeze().sum().item()
                    epoch_loss += loss.item()
                    total += batch_size
                    epoch_acc =  epoch_correct / total

                    if (i + 1) % print_freq == 0:
                        printed = f'<<< Valid: Epoch: [{epoch:03d}], Step: [{i+1:03d}/{total_loader}], loss: {loss.item():>6.4f}, acc: {epoch_acc:.4f}'
                        logger.info(printed)
                    pbar.set_description(f'Validate: Epoch: [{epoch:03d}], loss: {loss.item():.4f}, acc: {epoch_acc:.4f}')
                    pbar.update(1)
    
    train_acc = epoch_correct / total
    train_loss = epoch_loss / total_loader

    return train_acc, train_loss