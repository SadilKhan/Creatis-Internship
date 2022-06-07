from os.path import join
from turtle import forward
import numpy as np
import colorsys, random, os, sys
import pandas as pd
import torch 
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import *

class TotalLoss(nn.Module):
    def __init__(self,seg_loss_type="ce",weights=None,lamda1=10,lamda2=10):
        super(TotalLoss, self).__init__()
        self.seg_loss_type = seg_loss_type
        self.weights = weights
        self.lamda1 = lamda1
        self.lamda2 = lamda2

        if self.seg_loss_type=="ce":
            self.seg_loss=CrossEntropy(self.weights)
        self.sdfLoss=AAAI_sdf_loss()
        self.saLoss= SALoss()


    def forward(self,points,latent,predSeg,trueSeg,predSDF,trueSDF):
        self.loss1=self.seg_loss(predSeg,trueSeg)
        self.loss2=self.sdfLoss(predSDF,trueSDF)
        self.loss3=self.saLoss(points,latent,trueSeg)





# Loss for Segmentation
class CrossEntropy(nn.Module):
    def __init__(self,weights=None):
        super(CrossEntropy, self).__init__()

        self.weights=weights
        self.ce=nn.CrossEntropy(weights=self.weights)
    
    def forward(self,pred,true):
        loss=self.ce(pred,true)
        return loss

def calculate_weight(ratio,alpha1=0.3,alpha2=0.3,alpha3=0.4):
    weight= 0.3+0.3*(1/(ratio+0.02))+0.4*(1/ratio)
    return weight


class FPCrossEntropyLoss(nn.Module):
    """ Cross Entopy Loss which takes into account the background error"""
    def __init__(self,ratio_type="inverse"):
        super(FPCrossEntropyLoss, self).__init__()
        self.ratio_type = ratio_type

    def forward(self,pred,true):
        """
        pred: shape (B,C,N)
        true: shape (B,N)
        """
        pred=F.softmax(pred,dim=1)
        M=pred.size(1)
        loss=0
        freq=torch.einsum("bch->bh", F.one_hot(true)).type(torch.float32)
        ratio=freq/torch.sum(freq)
        if self.ratio_type == "inverse":
            w=1/ratio
        else:
            w=calculate_weight(ratio)
        w_sum=0
        for i in range(M):
            pos_i=(true==i).nonzero()[:,1]
            pred_i=pred[:,:,pos_i]
            pred_label_i=pred_i.argmax(dim=1) # The prediction Label
            pred_label_i[:,1:]=pred_label_i[:,1:]/2
            if i==0:
                weight_i=w[:,pred_label_i[0]]
                w_sum+=torch.sum(weight_i)
            else:
                weight_i=w[:,i]
                w_sum+=torch.sum(weight_i*freq[:,i])

            loss+=torch.sum(-1*weight_i*torch.log(pred_i[:,i,:]))
    
        return loss/w_sum



# Loss for Signed Distance Function
def AAAI_sdf_loss(net_output, gt_sdm):
    """ From https://github.com/JunMa11/SegWithDistMap/blob/master/code/train_LA_AAAISDF.py"""
    # print('net_output.shape, gt_sdm.shape', net_output.shape, gt_sdm.shape)
    # ([1,1,100000])
    net_output=net_output[:,0]
    smooth = 1e-5
    # compute eq (4)
    intersect = torch.sum(net_output * gt_sdm)
    pd_sum = torch.sum(net_output ** 2)
    gt_sum = torch.sum(gt_sdm ** 2)
    L_product = (intersect + smooth) / (intersect + pd_sum + gt_sum + smooth)
    # print('L_product.shape', L_product.shape) (4,2)
    L_SDF_AAAI = - L_product + torch.norm(net_output - gt_sdm, 1)/torch.numel(net_output)

    return L_SDF_AAAI

# Loss for Latent Vector                                                                        
class SALoss(nn.Module):
    def __init__(self,alpha=0.7,beta=1.5,weight=[],aggregate="mean"):
        super().__init__()
        self.alpha=0.7
        self.beta=1.5
        self.weight=weight
        self.aggregate=aggregate


    def forward(self,points,embedding,true):
        """
            points: shape (B,N,3)
            embedding: shape (B,N,k)
            true: shape (B,N,C)
        
        """
        M=len(torch.unique(true)) # Number of Labels
        N=points.size(1) # Number of points
        intraLoss=0 # Loss for Similar points
        interLoss=0 # Loss for Different Points
        mean_emb=[] # Mean embeddings for organs
        if len(self.weight)==0:
            self.weight=[1]*M

        # Calculate Intra Loss
        for i in range(1,M):
            pos=(true==i).nonzero()[:,1]
            emb_i=embedding[:,pos] # (B,N1,k)
            point_i=points[:,pos] # (B,N1,3)
            g=torch.sigmoid(torch.sum(point_i**2,dim=-1)**0.5) # points are already normalized (B,N1)
            mean_emb_i=torch.mean(embedding[:,pos],dim=1) # (B,k)
            mean_emb.append(mean_emb_i)
            if self.aggregate=="mean":
                intraLoss+=torch.mean(self.weight[i]*g*torch.square(torch.clamp(torch.sum((emb_i-mean_emb_i)**2,dim=-1)**0.5-self.alpha,min=0)))
            else:
                intraLoss+=torch.sum(self.weight[i]*g*torch.square(torch.clamp(torch.sum((emb_i-mean_emb_i)**2,dim=-1)**0.5-self.alpha,min=0)))
        mean_emb=torch.stack(mean_emb,dim=1) #(B,N2,k)

        # Calculate Inter Loss
        for i in range(1,M):
            for j in range(1,M):
                if (i!=j):
                    interLoss+=torch.square(torch.clamp(self.beta-torch.sum((mean_emb[:,i-1]-mean_emb[:,j-1])**2,dim=-1)**0.5,min=0))
        
        return intraLoss/M+interLoss/(M*(M-1))

