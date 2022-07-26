import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

def accuracy(scores, labels):
    r"""
        Compute the per-class accuracies and the overall accuracy # TODO: complete doc

        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels

        Returns
        -------
        list of floats of length num_classes+1 (last item is overall accuracy)
    """
    num_classes = len(torch.unique(labels)) # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = torch.max(scores, dim=-2).indices
    #predictions=(scores>0.5)*1

    accuracies = []

    accuracy_mask = predictions == labels
    for label in range(num_classes):
        label_mask = labels == label
        per_class_accuracy = (accuracy_mask & label_mask).float().sum()
        per_class_accuracy /= label_mask.float().sum()
        accuracies.append(per_class_accuracy.cpu().item())
    # overall accuracy
    accuracies.append(accuracy_mask.float().mean().cpu().item())
    return accuracies

def calculate_precision(scores,labels):
    """Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels"""

    predLabel=scores.argmax(dim=1)[0]
    #predLabel=(scores[0]>0.5)*1
    labels=labels[0]
    precisions=precision_score(labels.cpu(),predLabel.cpu(),average=None,zero_division=0)
    #print(set(labels.tolist())-set(predLabel.tolist()))

    return precisions


def intersection_over_union(scores, labels):
    r"""
        Compute the per-class IoU and the mean IoU # TODO: complete doc

        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels

        Returns
        -------
        list of floats of length num_classes+1 (last item is mIoU)
    """
    num_classes = len(torch.unique(labels)) # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = torch.max(scores, dim=-2).indices
    #predictions=(scores>0.5)*1

    ious = []

    for label in range(num_classes):
        pred_mask = predictions == label
        labels_mask = labels == label
        iou = (pred_mask & labels_mask).float().sum() / (pred_mask | labels_mask).float().sum()
        ious.append(iou.cpu().item())
    ious.append(np.nanmean(ious))
    return ious

def plot(array1,array2,legend1,legend2,title,path):
    epochs=list(range(len(array1)))

    fig=plt.figure(figsize=(10,10))
    plt.plot(epochs,array1)
    plt.plot(epochs,array2)
    plt.legend([legend1,legend2])
    plt.title(title)
    plt.savefig(path+"/"+legend1.split(" ")[-1]+".jpg",dpi=600)
