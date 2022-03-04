import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from model import Model, transformer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from model_utils import *
from sklearn.metrics import accuracy_score
from tqdm import tqdm

sys.path.append("/home/khan/Internship/Codes/Datagen")
from dataset import PointCloudDataset



def train(args,num_classes=8):

    train_loader=DataLoader(PointCloudDataset(args.csvPath,
    transforms=args.transforms,mode="train"),batch_size=args.batch_size)

    test_loader=DataLoader(PointCloudDataset(args.csvPath,
    transforms=False,mode="test"),batch_size=args.batch_size)


    LR=args.lr  
    EPOCHS=args.epochs
    device=torch.device("cuda")

    model=Model(args,transformer,num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)


    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr / 100)


    for epoch in range(EPOCHS):
        train_loss=0
        model.train()
        train_pred = []
        train_true = []
        train_iou=[]
        for batch_data in tqdm(train_loader, total=len(train_loader)):
            data, label = batch_data
            data, label = data.to(device), label.to(device).squeeze()
            batch_size = data.shape[0]
            # start training the model
            opt.zero_grad()
            logits = model(data)


            loss,iou = cal_loss(logits, label)
            loss.backward()
            opt.step()

            preds = logits.max(dim=1)[1]
            train_true.append(label.detach().cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            train_iou.append(iou)
            count += batch_size
            train_loss += loss.item() * batch_size
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        epoch_loss = train_loss * 1.0 / count
        train_acc = accuracy_score(train_true, train_pred)
        #train_bal_acc = balanced_accuracy_score(train_true, train_pred)
        best_iou=np.mean(train_iou)

        scheduler.step()
    torch.save(model.state_dict(),"/home/khan/Internship/Codes/model2.pth") 
    print(f"Epoch {epoch}, loss: {loss}, iou: {iou}, train_acc: {train_acc}")                                                                              



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--csvPath', type=str,
                        help='CSV File Directory')
    parser.add_argument('--transforms', type=bool,default=True,
                        help='CSV File Directory')                    
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N', choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--use_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--use_gpus', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--use_norm', type=bool, default=False,
                        help='Whether to use norm')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--num_K', nargs='+', type=int,
                        help='list of num of neighbors')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--head', type=int, default=8, metavar='N',
                        help='Dimension of heads')
    parser.add_argument('--dim_k', type=int, default=32, metavar='N',
                        help='Dimension of key/query tensors')
    args = parser.parse_args()

    train(args,8)
    

