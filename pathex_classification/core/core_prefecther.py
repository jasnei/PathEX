import time
import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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

def train(model, train_loader, device, loss_fn, optimizer, epoch):

    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    model.train()

    # start_time = time.time()
    prefetcher = data_prefetcher(train_loader)
    images, labels = prefetcher.next()

    i = 0
    acc_list = []
    loss_list = []
    total = 0
    correct = 0
    while images is not None:
        i += 1
        start_time = time.time()

        # Forward
        images = images.to(device)     
        preds = model(images)

        # compute loss
        loss = loss_fn(preds, labels) 

        # Backward        
        optimizer.zero_grad()                     
        loss.backward()

        # update weights
        optimizer.step()

        # prediction -> acc
        _, pred_labels = torch.max(preds, 1)
        batch_size = labels.size(0)
        batch_correct = (pred_labels==labels).squeeze().sum().item()
        batch_acc = batch_correct / batch_size

        # correct += batch_correct
        # total += batch_size

        loss = loss.cpu().detach().numpy()
        acc_list.append(batch_acc)
        loss_list.append(loss)
        elapse = time.time() - start_time
        print("epoch: {}, step: {}, acc: {:.4f}, loss: {:.4f}, elapse: {:.4f}".format(epoch, i, batch_acc, loss, elapse))

        images, labels = prefetcher.next()
    return acc_list, loss_list
    

def valid(model, val_loader, device, loss_fn, optimizer, epoch):
    correct = 0
    total = 0
    loss_val = 0
    model.eval()
    with torch.no_grad():
        acc_list = []
        loss_list = []
        for i, (images, labels) in enumerate(val_loader):
            batch_correct = 0
            batch_loss = 0
            batch_size = labels.size(0)

            images = images.to(device)
            labels = labels.to(device).long()

            preds = model(images)
            loss = loss_fn(preds, labels)

            _, pred_labels = torch.max(preds, 1)
            
            batch_correct = (pred_labels==labels).squeeze().sum().item()
            batch_loss = loss.item()
            batch_acc = batch_correct / batch_size

            acc_list.append(batch_acc)
            loss_list.append(batch_loss)

            correct += batch_correct
            loss_val += batch_loss
            total += batch_size
            print('batch: {}, batch_acc: {:.4f}, batch_loss: {:.4f}'.format(i, batch_acc, batch_loss))

    val_acc = correct / total
    # print('epoch: {}, Val_acc: {:.4f}, val_loss: {:.4f}, total: {}'.format(epoch, val_acc, loss_val, total))
    return acc_list, loss_list, val_acc



def train_prefetch(model, train_loader, loss_fn, optimizer, train_total, epoch):

    model.train()

    train_loss = 0.
    train_acc = 0.
    total = 0

    # start_time = time.time()
    prefetcher = data_prefetcher(train_loader)
    images, labels = prefetcher.next()

    iteration = 0
    while images is not None:
        iteration += 1

        # Forward
        images = images.cuda()
        labels = labels.long().cuda()     
        preds = model(images)
        # preds = torch.nn.functional.softmax(preds, dim=-1)

        # Compute loss
        loss = loss_fn(preds, labels) 
        train_loss += loss.item()
    
        # Backward        
        optimizer.zero_grad()                     
        loss.backward()

        # Update weights
        optimizer.step()

        # Prediction -> acc
        _, pred_labels = torch.max(preds, 1)
        # pred_labels = preds.squeeze()
        batch_correct = (pred_labels==labels).squeeze().sum().item()
        train_acc += batch_correct

        batch_size = labels.size(0)
        total += batch_size

        # correct += batch_correct
        # total += batch_size

        # y_true.extend(labels.cpu().numpy())
        # y_pred.extend(pred_labels.cpu().numpy())
        images, labels = prefetcher.next()
    train_acc = train_acc / train_total
    train_loss = train_loss / len(train_loader)
    
    return train_acc, train_loss #, y_pred, y_true
    
def valid_prefecth(model, val_loader, loss_fn, optimizer, valid_total, epoch):
    valid_acc = 0.
    valid_loss = 0.
    # y_true = []
    # y_pred = []

    # start_time = time.time()
    prefetcher = data_prefetcher(val_loader)
    images, labels = prefetcher.next()

    model.eval()
    with torch.no_grad():
        iteration = 0
        while images is not None:
            iteration += 1

            images = images.cuda()
            labels = labels.long().cuda()

            preds = model(images)

            loss = loss_fn(preds, labels)
            valid_loss += loss.item()

            _, pred_labels = torch.max(preds, 1)
            batch_correct = (pred_labels==labels).squeeze().sum().item()
            valid_acc += batch_correct

            # y_true.extend(labels.cpu().numpy())
            # y_pred.extend(pred_labels.cpu().numpy())
            images, labels = prefetcher.next()
    valid_acc = valid_acc / valid_total
    valid_loss = valid_loss / len(val_loader)

    return valid_acc, valid_loss #, y_pred, y_true

