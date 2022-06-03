import sys
sys.path.append("/home/khan/Internship/Codes/Datagen")
from model_utils.metrics import *
from model_utils.tools import Config as cfg
from model_utils.tools import *
from torch.utils.data import DataLoader
from model_test_run import RandLANet
from dataset import PointCloudDataset
import torch.nn.functional as F
import torch.nn as nn
import torch
import warnings
from tqdm import tqdm
import argparse
from datetime import datetime
import json
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
import time
from sklearn.model_selection import KFold


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    accuracies = []
    ious = []
    with torch.no_grad():
        for points, labels in tqdm(loader, desc='Validation', leave=False):
            points = points.to(device)
            labels = labels.to(device)
            scores,embeddings = model(points)
            # probScores = nn.Softmax(dim=1)(scores)
            loss = criterion(scores, labels,embeddings)
            losses.append(loss.cpu().item())
            accuracies.append(accuracy(scores, labels))
            ious.append(intersection_over_union(scores, labels))
    return np.mean(losses), np.nanmean(np.array(accuracies), axis=0), np.nanmean(np.array(ious), axis=0)


def train(args,csvFiles, train_transforms=False, load=False,fold_num=0):
    global Path
    logs_dir = args.logs_dir / args.name
    logs_dir.mkdir(exist_ok=True, parents=True)
    

    if args.expr_type == "smote":
        train_loader = DataLoader(PointCloudDataset(
            csvFiles["train"], transforms=train_transforms, mode="train", expr_type="smote"), batch_size=args.batch_size)
    elif args.expr_type=="adasyn":
        train_loader = DataLoader(PointCloudDataset(
            csvFiles["train"], transforms=train_transforms, mode="train", expr_type="adasyn"), batch_size=args.batch_size)
    else:
        train_loader = DataLoader(PointCloudDataset(
            csvFiles["train"], transforms=train_transforms, mode="train"), batch_size=args.batch_size)

    val_loader = DataLoader(PointCloudDataset(
            csvFiles["test"], transforms=False, mode="test"), batch_size=args.batch_size)

    d_in = next(iter(train_loader))[0].size(-1)
    num_classes = 6
    model = RandLANet(
        d_in,
        num_classes=num_classes,
        num_neighbors=args.neighbors,
        decimation=args.decimation,
        device=args.gpu
    )
    if args.dataset_type=="visceral":
        n_samples = torch.tensor(cfg.class_weights_visceral, dtype=torch.float, device=args.gpu)
    elif args.dataset_type=="ctorg":
        n_samples= torch.tensor(cfg.class_weights_ctorg,dtype=torch.float,device=args.gpu)
    ratio_samples = n_samples / n_samples.sum()
    weights = 1 / (ratio_samples)
    criterion = PcLossTEST(ratio_samples)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.adam_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, args.scheduler_gamma)

    first_epoch = 1
    Path = "/home/khan/Internship/Codes/modelv1.pth"
    if load:
        model.load_state_dict(torch.load(Path))
    
    for epoch in range(first_epoch, args.epochs+1):
        output=open("output.txt",mode="a+")
        print(f'=== EPOCH {epoch:d}/{args.epochs:d} ===',file=output)
        t0 = time.time()
        # Train
        model.train()

        # metrics
        losses = []

        accuracies = []
        ious = []

        all_training_ious = []
        all_val_ious = []
        all_training_losses = []
        all_val_losses = []

        # iterate over dataset
        for points, labels in tqdm(train_loader, desc='Training', leave=False):
            points = points.to(args.gpu)
            labels = labels.to(args.gpu)
            optimizer.zero_grad()
            scores,embedding = model(points)
            loss = criterion(scores, labels,embedding)
 

            loss.backward()

            optimizer.step()

            losses.append(loss.cpu().item())
            accuracies.append(accuracy(scores, labels))
            ious.append(intersection_over_union(scores, labels))

        scheduler.step()

        accs = np.nanmean(np.array(accuracies), axis=0)
        ious = np.nanmean(np.array(ious), axis=0)

        val_loss, val_accs, val_ious = evaluate(
            model,
            val_loader,
            criterion,
            args.gpu
        )
        all_val_losses.append(np.mean(val_loss))
        all_training_losses.append(np.mean(losses))

        all_training_ious.append(np.mean(ious))
        all_val_ious.append(np.mean(val_ious))

        loss_dict = {
            'Training loss':    np.mean(losses),
            'Validation loss':  val_loss
        }
        acc_dicts = [
            {
                'Training accuracy': acc,
                'Validation accuracy': val_acc
            } for acc, val_acc in zip(accs, val_accs)
        ]
        iou_dicts = [
            {
                'Training accuracy': iou,
                'Validation accuracy': val_iou
            } for iou, val_iou in zip(ious, val_ious)
        ]

        t1 = time.time()
        d = t1 - t0
        # Display results
        
        for k, v in loss_dict.items():
            print(f'{k}: {v:.7f}', end='\t',file=output)
        print(file=output)
        print('Accuracy     ', *
            [f'{i:>5d}' for i in range(num_classes)], '   OA', sep=' | ',file=output)
        print('Training:    ', *[f'{acc:.3f}' if not np.isnan(acc)
            else '  nan' for acc in accs], sep=' | ',file=output)
        print('Validation:  ', *[f'{acc:.3f}' if not np.isnan(acc)
            else '  nan' for acc in val_accs], sep=' | ',file=output)

        print('IoU          ', *
            [f'{i:>5d}' for i in range(num_classes)], ' mIoU', sep=' | ',file=output)
        print('Training:    ', *[f'{iou:.3f}' if not np.isnan(iou)
            else '  nan' for iou in ious], sep=' | ',file=output)
        print('Validation:  ', *[f'{iou:.3f}' if not np.isnan(iou)
            else '  nan' for iou in val_ious], sep=' | ',file=output)

        print('Time elapsed:', '{:.0f} s'.format(
            d) if d < 60 else '{:.0f} min {:02.0f} s'.format(*divmod(d, 60)),file=output)
        torch.save(model.state_dict(), Path)
        output.close()
    return model


if __name__ == '__main__':

    """Parse program arguments"""
    parser = argparse.ArgumentParser(
        prog='RandLA-Net',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    base = parser.add_argument_group('Base options')
    expr = parser.add_argument_group('Experiment parameters')
    param = parser.add_argument_group('Hyperparameters')
    dirs = parser.add_argument_group('Storage directories')
    misc = parser.add_argument_group('Miscellaneous')

    base.add_argument('--csvPath', type=str, help='location of the dataset',
                      default='/home/khan/Internship/dataset/point_cloud')

    expr.add_argument('--epochs', type=int, help='number of epochs',
                      default=50)
    expr.add_argument('--load', type=str, help='model to load',
                      default='')
    expr.add_argument('--dataset_type',type=str,default="visceral")
    expr.add_argument('--expr_type', type=str,
                      help="Experiment Type - wm/smote/adasyn", default="wm")

    param.add_argument('--adam_lr', type=float, help='learning rate of the optimizer',
                       default=1e-2)
    param.add_argument('--batch_size', type=int, help='batch size',
                       default=1)
    param.add_argument('--decimation', type=int, help='ratio the point cloud is divided by at each layer',
                       default=4)
    param.add_argument('--dataset_sampling', type=str, help='how dataset is sampled',
                       default='active_learning', choices=['active_learning', 'naive'])
    param.add_argument('--neighbors', type=int, help='number of neighbors considered by k-NN',
                       default=16)
    param.add_argument('--scheduler_gamma', type=float, help='gamma of the learning rate scheduler',
                       default=0.95)

    dirs.add_argument('--test_dir', type=str, help='location of the test set in the dataset dir',
                      default='test')
    dirs.add_argument('--train_dir', type=str, help='location of the training set in the dataset dir',
                      default='train')
    dirs.add_argument('--val_dir', type=str, help='location of the validation set in the dataset dir',
                      default='val')
    dirs.add_argument('--logs_dir', type=Path, help='path to tensorboard logs',
                      default='runs')

    misc.add_argument('--gpu', type=int, help='which GPU to use (-1 for CPU)',
                      default=0)
    misc.add_argument('--name', type=str, help='name of the experiment',
                      default=None)
    misc.add_argument('--num_workers', type=int, help='number of threads for loading data',
                      default=0)
    misc.add_argument('--save_freq', type=int, help='frequency of saving checkpoints',
                      default=10)

    args = parser.parse_args()

    if args.gpu >= 0:
        if torch.cuda.is_available():
            args.gpu = torch.device(f'cuda:{args.gpu:d}')
        else:
            warnings.warn(
                'CUDA is not available on your machine. Running the algorithm on CPU.')
            args.gpu = torch.device('cpu')
    else:
        args.gpu = torch.device('cpu')

    if args.name is None:
        if args.load:
            args.name = args.load
        else:
            args.name = datetime.now().strftime('%Y-%m-%d_%H:%M')

    t0 = time.time()
 

    # 5 fold cross Validation dataset
    csvFiles=np.array(glob(os.path.join(args.csvPath,"*")))
    #print(csvFiles)
    splitDataset=dict()
    for i in range(5):
        splitDataset[i]=dict()

    kf=KFold(n_splits=5)
    i=0
    for train_index, test_index in kf.split(csvFiles):
        x_train,x_test=csvFiles[train_index],csvFiles[test_index]
        splitDataset[i]['train']=list(x_train)
        splitDataset[i]['test']=list(x_test)
        i+=1
    Path = "/home/khan/Internship/Codes/modelv1.pth"
    for i in range(5):
        model = train(args,splitDataset[i],train_transforms=False,load=False,fold_num=i)
        print(f"FOLD {i} COMPLETE\n")
        torch.save(model.state_dict(), Path)
    t1 = time.time()

    d = t1 - t0
    print('Done. Time elapsed:', '{:.0f} s.'.format(
        d) if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))
