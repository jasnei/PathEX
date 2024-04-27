import time
import torch
import numpy as np


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


def train(model, train_loader, loss_fn, optimizer, train_total, epoch):

    model.train()

    train_loss = 0.
    train_acc = 0.
    total = 0
    # y_true = []
    # y_pred = []
    for i, (images, labels) in enumerate(train_loader):

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

        # y_true.extend(labels.cpu().numpy())
        # y_pred.extend(pred_labels.cpu().numpy())
    train_acc = train_acc / train_total
    train_loss = train_loss / len(train_loader)
    
    return train_acc, train_loss #, y_pred, y_true
    
def valid(model, val_loader, loss_fn, optimizer, valid_total, epoch):
    valid_acc = 0.
    valid_loss = 0.
    # y_true = []
    # y_pred = []
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):

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
    valid_acc = valid_acc / valid_total
    valid_loss = valid_loss / len(val_loader)

    return valid_acc, valid_loss #, y_pred, y_true


def valid_test_set(model, val_loader,  valid_total):
    valid_acc = 0.
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:

            images = images.cuda()
            labels = labels.long().cuda()

            preds = model(images)

            _, pred_labels = torch.max(preds, 1)
            batch_correct = (pred_labels==labels).squeeze().sum().item()
            valid_acc += batch_correct

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred_labels.cpu().numpy())
    valid_acc = valid_acc / valid_total

    return valid_acc, y_true, y_pred

def batch_inference(model, device, val_loader):
    preds = []
    probs = []
    for images in val_loader:

        # images = images.cuda()
        images = images.to(device)
        outs = model(images)

        outs = torch.nn.functional.softmax(outs, dim=1)
        prob, index = torch.max(outs, 1)

        preds.extend(index.cpu().numpy())
        probs.extend(prob.cpu().numpy())

    return preds, probs
