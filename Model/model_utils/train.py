import argparse
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import time
import sys
sys.path.append("/home/khan/Internship/Codes/Datagen")
from tqdm import tqdm
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import PointCloudDataset
from model import RandLANet
from torch.utils.data import DataLoader
from model_utils.tools import Config as cfg
from model_utils.metrics import accuracy, intersection_over_union

def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    accuracies = []
    ious = []
    with torch.no_grad():
        for points, labels in tqdm(loader, desc='Validation', leave=False):
            points = points.to(device)
            labels = labels.to(device)
            scores = model(points)
            probScores=nn.Softmax(dim=1)(scores)
            loss = criterion(probScores, labels)
            losses.append(loss.cpu().item())
            accuracies.append(accuracy(scores, labels))
            ious.append(intersection_over_union(scores, labels))
    return np.mean(losses), np.nanmean(np.array(accuracies), axis=0), np.nanmean(np.array(ious), axis=0)


def train(args,train_transforms=True,load=False):
    global Path
    logs_dir = args.logs_dir / args.name
    logs_dir.mkdir(exist_ok=True, parents=True)

    train_loader=DataLoader(PointCloudDataset(args.csvPath,
    transforms=train_transforms,mode="train"),batch_size=args.batch_size)

    val_loader=DataLoader(PointCloudDataset(args.csvPath,
    transforms=False,mode="test"),batch_size=args.batch_size)

    d_in = next(iter(train_loader))[0].size(-1)
    num_classes=8
    model = RandLANet(
        d_in,
        num_classes=8,
        num_neighbors=args.neighbors,
        decimation=args.decimation,
        device=args.gpu
    )

    #print('Computing weights...', end='\t')
    samples_per_class = np.array(cfg.class_weights)

    n_samples = torch.tensor(cfg.class_weights, dtype=torch.float, device=args.gpu)
    ratio_samples = n_samples / n_samples.sum()
    weights = 1 / (ratio_samples)

    #print('Done.')
    #print('Weights:', weights)
    criterion = nn.CrossEntropyLoss(weight=weights)
    #criterion=nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.adam_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.scheduler_gamma)

    first_epoch = 1
    if load:
        model.load_state_dict(torch.load(Path))

    for epoch in range(first_epoch, args.epochs+1):
        print(f'=== EPOCH {epoch:d}/{args.epochs:d} ===')
        t0 = time.time()
        # Train
        model.train()

        # metrics
        losses = []
        accuracies = []
        ious = []

        # iterate over dataset
        for points, labels in tqdm(train_loader, desc='Training', leave=False):
            points = points.to(args.gpu)
            labels = labels.to(args.gpu)
            optimizer.zero_grad()
            scores = model(points)
            probScores=nn.Softmax(dim=1)(scores)
            #print(probScores)


            #logp = torch.distributions.utils.probs_to_logits(scores, is_binary=False)
            loss = criterion(probScores, labels)
            # logpy = torch.gather(logp, 1, labels)
            # loss = -(logpy).mean()

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
            print(f'{k}: {v:.7f}', end='\t')
        print()

        print('Accuracy     ', *[f'{i:>5d}' for i in range(num_classes)], '   OA', sep=' | ')
        print('Training:    ', *[f'{acc:.3f}' if not np.isnan(acc) else '  nan' for acc in accs], sep=' | ')
        print('Validation:  ', *[f'{acc:.3f}' if not np.isnan(acc) else '  nan' for acc in val_accs], sep=' | ')

        print('IoU          ', *[f'{i:>5d}' for i in range(num_classes)], ' mIoU', sep=' | ')
        print('Training:    ', *[f'{iou:.3f}' if not np.isnan(iou) else '  nan' for iou in ious], sep=' | ')
        print('Validation:  ', *[f'{iou:.3f}' if not np.isnan(iou) else '  nan' for iou in val_ious], sep=' | ')

        print('Time elapsed:', '{:.0f} s'.format(d) if d < 60 else '{:.0f} min {:02.0f} s'.format(*divmod(d, 60)))

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
            warnings.warn('CUDA is not available on your machine. Running the algorithm on CPU.')
            args.gpu = torch.device('cpu')
    else:
        args.gpu = torch.device('cpu')

    if args.name is None:
        if args.load:
            args.name = args.load
        else:
            args.name = datetime.now().strftime('%Y-%m-%d_%H:%M')

    t0 = time.time()
    Path="/home/khan/Internship/Codes/modelv1.pth"
    for i in range(5):
        if i==0:
            model=train(args,False)
        else:
            model=train(args,True,True)
        torch.save(model.state_dict(),Path)
    t1 = time.time()

    d = t1 - t0
    print('Done. Time elapsed:', '{:.0f} s.'.format(d) if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))
