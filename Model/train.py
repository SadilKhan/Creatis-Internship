import sys
sys.path.append("/home/khan/Internship/Codes/Datagen")
from model_utils.metrics import *
#from model_utils.dice_loss import *
from model_utils.tools import Config as cfg
from model_utils.tools import *
from torch.utils.data import DataLoader
from model import RandLANet
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
from sklearn.model_selection import KFold,train_test_split


def evaluate(model, loader, criterion, device,output):
    model.eval()
    losses = []
    accuracies = []
    ious = []
    precisions =[]
    with torch.no_grad():
        for name,points,nbr_intensity,labels,sdf in tqdm(loader, desc='Validation', leave=False):
            points = points.to(device)
            labels = labels.to(device)
            sdf=sdf.to(device)
            nbr_intensity=nbr_intensity.to(device)
            scores,predSdf,_,lv,permutation = model(points,nbr_intensity)
            loss = criterion(scores,labels.float(),predSdf,sdf)
            losses.append(loss.cpu().item())
            accuracies.append(accuracy(scores, labels))
            ious.append(intersection_over_union(scores, labels))
            precisions.append(calculate_precision(scores,labels))
            #print(f"TEST\nNAME:{name}\nLOSS:{loss}\nIOU:{intersection_over_union(scores, labels)}\nACC:{accuracy(scores, labels)}",file=output)
    return np.mean(losses), np.nanmean(np.array(accuracies), axis=0), np.nanmean(np.array(ious), axis=0),np.nanmean(np.array(precisions), axis=0)


def train(args,csvFiles, train_transforms=False, load=False):
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

    d_in = next(iter(train_loader))[1].size(-1)
    num_classes = 2
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
    criterion=TotalLoss()
    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.adam_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, args.scheduler_gamma)

    first_epoch = 1
    #for name,val in model.named_parameters():
    #print(name,"\t",val.requires_grad)
    
    for epoch in range(first_epoch, args.epochs+1):
        output=open(f"{args.model_name}.txt",mode="a+")
        print(f'=== EPOCH {epoch:d}/{args.epochs:d} ===',file=output)
        t0 = time.time()
        # Train
        model.train()

        # metrics
        losses = []

        accuracies = []
        ious = []
        precisions=[]

        all_training_ious = []
        all_val_ious = []
        all_training_losses = []
        all_val_losses = []

        # iterate over dataset
        for name,points,nbr_intensity,labels,sdf in tqdm(train_loader, desc='Training', leave=False):
            points = points.to(args.gpu)
            labels = labels.to(args.gpu)
            nbr_intensity=nbr_intensity.to(args.gpu)
            sdf=sdf.to(args.gpu)
            optimizer.zero_grad()
            scores,predSdf,_,lv,permutation = model(points,nbr_intensity)
            #print(outScores.shape)

            loss = criterion(scores,labels.float(),predSdf,sdf)
            loss.backward()
            #print(model.sdf_output.conv.weight)
            optimizer.step()
            losses.append(loss.cpu().item())
            accuracies.append(accuracy(scores, labels))
            ious.append(intersection_over_union(scores, labels))
            precisions.append(calculate_precision(scores,labels))
            #print(f"TRAIN\nNAME:{name}\nLOSS:{loss}\nIOU:{intersection_over_union(scores, labels)}\nACC:{accuracy(scores, labels)}",file=output)

        scheduler.step()
        print("\n\n",file=output)

        accs = np.nanmean(np.array(accuracies), axis=0)
        ious = np.nanmean(np.array(ious), axis=0)
        precisions = np.nanmean(np.array(precisions), axis=0)
        #print(precisions,ious)
        

        val_loss, val_accs, val_ious,val_precisions = evaluate(
            model,
            val_loader,
            criterion,
            args.gpu,output
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
                'Training Recall': acc,
                'Validation Recall': val_acc
            } for acc, val_acc in zip(accs, val_accs)
        ]
        iou_dicts = [
            {
                'Training IOU': iou,
                'Validation IOU': val_iou
            } for iou, val_iou in zip(ious, val_ious)
        ]

        precision_dicts=[
            {
                'Training Precision': precisions,
                'Validation Precision': val_precisions
            } for pr, val_fr in zip(precisions, val_precisions)
        ]

        t1 = time.time()
        d = t1 - t0
        # Display results
        
        for k, v in loss_dict.items():
            print(f'{k}: {v:.7f}', end='\t',file=output)
        print(file=output)
        print('Recall       ', *
            [f'{i:>5d}' for i in range(num_classes)], ' OR', sep=' | ',file=output)
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

        print('Precision    ', *
            [f'{i:>5d}' for i in range(num_classes)], ' mPrecision', sep=' | ',file=output)
        print('Training:    ', *[f'{pr:.3f}' if not np.isnan(pr)
            else '  nan' for pr in precisions], sep=' | ',file=output)
        print('Validation:  ', *[f'{pr:.3f}' if not np.isnan(pr)
            else '  nan' for pr in val_precisions], sep=' | ',file=output)

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
                      default='--csvPath /home/khan/Internship/dataset/Extracted/point_cloud/visceral_7/')

    expr.add_argument('--epochs', type=int, help='number of epochs',
                      default=300)
    expr.add_argument('--load', type=str, help='model to load',
                      default='')
    expr.add_argument('--dataset_type',type=str,default="visceral")
    expr.add_argument('--expr_type', type=str,
                      help="Experiment Type - wm/smote/adasyn", default="wm")

    param.add_argument('--adam_lr', type=float, help='learning rate of the optimizer',
                       default=0.0001)
    param.add_argument('--batch_size', type=int, help='batch size',
                       default=1)
    param.add_argument('--decimation', type=int, help='ratio the point cloud is divided by at each layer',
                       default=2)
    param.add_argument('--dataset_sampling', type=str, help='how dataset is sampled',
                       default='active_learning', choices=['active_learning', 'naive'])
    param.add_argument('--neighbors', type=int, help='number of neighbors considered by k-NN',
                       default=8)
    param.add_argument('--scheduler_gamma', type=float, help='gamma of the learning rate scheduler',
                       default=0.95)
    param.add_argument("--nsplit",help="Cross Validation N Splits",type=int,default=5)

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
    misc.add_argument("--model_name",type=str,default="modelv1")

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
 

    # n fold cross Validation dataset
    Path = f"/home/khan/Internship/Codes/{args.model_name}.pth"
    csvFiles=np.array(glob(os.path.join(args.csvPath,"*")))
    splitDataset=dict()
    for i in range(args.nsplit):
        splitDataset[i]=dict()
    if args.nsplit==1:
        x_train,x_test=train_test_split(csvFiles,test_size=0.2,random_state=12)
        splitDataset[0]['train']=x_train
        splitDataset[0]['test']=x_test
        model = train(args,splitDataset[0],train_transforms=False,load=False)
        torch.save(model.state_dict(), Path)

    #print(csvFiles)
    else:
        kf=KFold(n_splits=args.nsplit)
        i=0
        for train_index, test_index in kf.split(csvFiles):
            x_train,x_test=csvFiles[train_index],csvFiles[test_index]
            splitDataset[i]['train']=list(x_train)
            splitDataset[i]['test']=list(x_test)
            i+=1
        for i in range(args.nsplit):
            dataFile=open("data.txt",mode="a+")
            print(f"FOLD{i}\nTRAIN:{splitDataset[i]['train']}\nTEST:{splitDataset[i]['test']}",file=dataFile)
            dataFile.close()
            model = train(args,splitDataset[i],train_transforms=False,load=False)
            print(f"FOLD {i+1} COMPLETE\n")
            torch.save(model.state_dict(), Path)

    t1 = time.time()

    d = t1 - t0
    print('Done. Time elapsed:', '{:.0f} s.'.format(
        d) if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))
