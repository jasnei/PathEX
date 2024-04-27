import os
import pickle
import torch

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # 在数据处理里已经做了标准化了，所以这里不再需要做了
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)

        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            # self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            # self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        # target = self.next_target
        self.preload()
        return input

def train_valid(model, loss_fn, optimizer, epoch, train_valid, log_path, dataloader, print_frq=100):

    if log_path and train_valid == 'train':
        log = open(os.path.join(log_path, "train_log.txt"), mode='a', encoding='utf-8')
    elif log_path and train_valid == 'valid':
        log = open(os.path.join(log_path, "valid_log.txt"), mode='a', encoding='utf-8')

    train_total = len(dataloader)
    train_loss = 0.
    #===================================================
    # Prefectcher
    #===================================================
    prefecther = data_prefetcher(dataloader)
    images = prefecther.next()
    
    i = 0
    if train_valid == "train":
        model.train()
        
        while images is not None:
        # for i, images in enumerate(dataloader):
            images = images.squeeze(0)#.cuda(non_blocking=True)
        
            #=========Forward============
            _, out = model(images)

            #===========Loss=============
            loss = loss_fn(out, images)

            if model.training:
                #==========Backward==========
                loss.backward()

                # Update weights
                optimizer.step()
                # Reset gradients, for the next accumulated batches
                optimizer.zero_grad()
            i += 1
            images = prefecther.next()
            train_loss += loss.item()
            if (i + 1) % print_frq == 0:
                # print(f"<<< Epoch: [{epoch+1:03d}], Step: [{i+1:03d}], pred_1: {pred_1.cpu().detach().numpy()}")
                print(f"<<< Training: Epoch: [{epoch:03d}], Step: [{i+1:05d}]/[{train_total}], loss: {loss.item():<7.4f}")
                # log.write(f"<<< Epoch: [{epoch+1:03d}], Step: [{i+1:03d}], pred_1: {pred_1.cpu().detach().numpy()}\n")
                log.write(f"<<< Training: Epoch: [{epoch:03d}], Step: [{i+1:05d}]/[{train_total}], loss: {loss.item():<7.4f}\n")
        
    elif train_valid == "valid":
        model.eval()
        with torch.no_grad():
            while images is not None:
            # for i, images in enumerate(dataloader):
                images = images.squeeze(0)#.cuda(non_blocking=True)
            
                #=========Forward============
                _, out = model(images)

                #===========Loss=============
                loss = loss_fn(out, images)

                if model.training:
                    #==========Backward==========
                    loss.backward()

                    # Update weights
                    optimizer.step()
                    # Reset gradients, for the next accumulated batches
                    optimizer.zero_grad()
                i += 1
                images = prefecther.next()
                train_loss += loss.item()
                if (i + 1) % print_frq == 0:
                    # print(f"<<< Epoch: [{epoch:03d}], Step: [{i+1:03d}], pred_1: {pred_1.cpu().detach().numpy()}")
                    print(f"<<< Valid: Epoch: [{epoch:03d}], Step: [{i+1:05d}]/[{train_total}], loss: {loss.item():<7.4f}")
                    # log.write(f"<<< Epoch: [{epoch:03d}], Step: [{i+1:03d}], pred_1: {pred_1.cpu().detach().numpy()}\n")
                    log.write(f"<<< Valid: Epoch: [{epoch:03d}], Step: [{i+1:05d}]/[{train_total}], loss: {loss.item():<7.4f}\n")

    train_loss = train_loss / train_total

    return train_loss