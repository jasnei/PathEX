import os
from time import time
import torch

def train_valid(model, loss_fn, optimizer, epoch, train_valid, log_path, loader):

    if log_path and train_valid == 'train':
        log = open(os.path.join(log_path, "train_log.txt"), mode='a', encoding='utf-8')
    elif log_path and train_valid == 'valid':
        log = open(os.path.join(log_path, "valid_log.txt"), mode='a', encoding='utf-8')

    total_train = len(loader)
    total_correct = 0
    train_acc = 0.
    train_loss = 0.

    if train_valid == "train":
        model.train()

        for i, fn, feature, label in enumerate(loader):
            start_time = time()
            file_name = os.path.abspath(fn).split(os.sep)[-1].split('.')[0]
            #=========Forward============
            pred = model(feature)

            #===========Loss=============
            loss = loss_fn(pred, label)

            if model.training:
                #==========Backward==========
                loss.backward()

                if (i+1) % 64 == 0 or (i+1) % 488 == 0:
                    # Update weights
                    optimizer.step()
                    # Reset gradients, for the next accumulated batches
                    optimizer.zero_grad()

            pred = torch.softmax(pred, dim=1)
            prob, pred_label = torch.max(pred, dim=1)

            total_correct += (pred_label==label).squeeze().sum().item()
            train_loss += loss.item()
            
            elapse = time() - start_time
            printed = f"<<< Training: Epoch: [{epoch+1:03d}], Step: [{i+1:03d}], Elapse: {elapse:4.4f}s, loss: {loss.item():>7.4f}, prob: {prob.item():.4f}, pred_label: {pred_label.item()}, label: {label.cpu().item()}"
            log_write = f"<<< Training: Epoch: [{epoch+1:03d}], Step: [{i+1:03d}], Elapse: {elapse:4.4f}s, file_name: {file_name:<41}, loss: {loss.item():>7.4f}, prob: {prob.item():.4f}, pred_label: {pred_label.item()}, label: {label.cpu().item()}\n"
            print(printed)
            log.write(log_write)
        
    elif train_valid == "valid":
        model.eval()
        with torch.no_grad():
            for i, fn, feature, label in enumerate(loader):
                start_time = time()
                file_name = os.path.abspath(fn).split(os.sep)[-1].split('.')[0]
                #=========Forward============
                pred = model(feature)

                #===========Loss=============
                loss = loss_fn(pred, label)

                pred = torch.softmax(pred, dim=1)
                prob, pred_label = torch.max(pred, dim=1)

                total_correct += (pred_label==label).squeeze().sum().item()
                train_loss += loss.item()

                elapse = time() - start_time
                printed = f"<<< Valid: Epoch: [{epoch+1:03d}], Step: [{i+1:03d}], Elapse: {elapse:4.4f}s, loss: {loss.item():>7.4f}, prob: {prob.item():.4f}, pred_label: {pred_label.item()}, label: {label.cpu().item()}"
                log_write = f"<<< Valid: Epoch: [{epoch+1:03d}], Step: [{i+1:03d}], Elapse: {elapse:4.4f}s, file_name: {file_name:<41}, loss: {loss.item():>7.4f}, prob: {prob.item():.4f}, pred_label: {pred_label.item()}, label: {label.cpu().item()}\n"
                print(printed)
                log.write(log_write)
    
                # # print(f"<<< Epoch: [{epoch+1:03d}], Step: [{i+1:03d}], pred_1: {pred_1.cpu().detach().numpy()}")
                # print(f"<<< Valid: Epoch: [{epoch+1:03d}], Step: [{i+1:03d}], loss: {loss.item():<7.4f}, prob: {prob.item():.4f}, pred_label: {pred_label.item()}, label: {label.cpu().item()}")
                # # log.write(f"<<< Epoch: [{epoch+1:03d}], Step: [{i+1:03d}], pred_1: {pred_1.cpu().detach().numpy()}\n")
                # log.write(f"<<< Valid: Epoch: [{epoch+1:03d}], Step: [{i+1:03d}], loss: {loss.item():<7.4f}, prob: {prob.item():.4f}, pred_label: {pred_label.item()}, label: {label.cpu().item()}\n")
        
    train_acc = total_correct / total_train
    train_loss = train_loss / total_train
    # print(f"<<< Epoch: [{epoch+1:03d}], train_acc: {train_acc:.4f}, train_loss: {train_loss:.4f}, ")

    return train_acc, train_loss