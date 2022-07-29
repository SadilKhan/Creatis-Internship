from os.path import join
from turtle import forward
import numpy as np
import colorsys, random, os, sys
import pandas as pd
import torch 
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import confusion_matrix
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

class Config:
    sampling_type = 'active_learning'
    #class_weights_visceral=[207194.95, 5628.1, 23958, 860.1, 1123.25, 1048.4]
    class_weights_visceral=[207194.95, 23958]
    class_weights_ctorg=[213092,5554,16651,610,1408,1419]

def heavisideFn(x,k=750,t=0):
    """ Approximation of Heaviside Function """
    return 1-(1/(1+torch.exp(-2*k*(x-t))))

def DiceLoss(pred,true,smooth=1e-7,weights=1):
    N=pred.shape[1]
    intersect=torch.sum(weights*pred*true)
    sumPred=torch.sum(weights*pred**2)
    sumTrue=torch.sum(true**2)
    diceLoss=1-2*(intersect+smooth)/(sumPred+sumTrue+smooth)
    
    return diceLoss

class PcLoss(nn.Module):
    def __init__(self,ratio):
        super().__init__()
        self.ratio=ratio
        self.loss1=nn.CrossEntropyLoss()
        self.loss2=nn.CrossEntropyLoss(weight=1/(self.ratio+0.02))
        self.loss3=nn.CrossEntropyLoss(weight=1/self.ratio)
    
    def forward(self,pred,true):
        value1=self.loss1(pred,true)
        value2=self.loss2(pred,true)
        value3=self.loss3(pred,true)

        value=0.3*value1+0.3*value2+0.4*value3

        return value


class SALoss(nn.Module):
    def __init__(self,alpha=0.7,beta=1.5,weight=[],aggregate="mean"):
        """ Structure Aware Loss Function """
        super().__init__()
        self.alpha=0.7
        self.beta=1.5
        self.weight=weight
        self.aggregate=aggregate


    def forward(self,points,embedding,true,permutation=[]):
        """
        points: shape (B,N,3)
        true: shape (B,N)
        embedding: shape (B,k,N)
        """
        #true=true.cpu()
        M=len(torch.unique(true)) # Number of Labels
        N=true.size(1) # Number of points in the original space
        N1=embedding.size(-1) # Number of points in the embedding space
        K=embedding.size(1)
        embedding=embedding.permute((0,2,1)).to("cuda") # (B,N,k)
        embedding.requires_grad=True
        intraLoss=0
        interLoss=0
        mean_emb=[] # Mean embeddings for organs
        if len(self.weight)==0:
            self.weight=[1]*M
        if len(permutation)>0:
            true=true[:,torch.argsort(permutation)][:,:N1]
            points=points[:,torch.argsort(permutation)][:,:N1]


        # Calculate Intra Loss
        for i in range(1,M):
            pos=(true==i).nonzero()[:,1]
            emb_i=embedding[:,pos] # (B,N1,k)
            point_i=points[:,pos] # (B,N1,3)
            mean_point_i=torch.mean(point_i,dim=1)
            g=torch.sigmoid(torch.sum((point_i-mean_point_i)**2,dim=-1)**0.5) # points are already normalized (B,N1)
            mean_emb_i=torch.mean(embedding[:,pos],dim=1).unsqueeze(-1) # (B,k)
            mean_emb.append(mean_emb_i)
            intraLoss_i=g*(1-torch.cosine_similarity(emb_i.permute((0,2,1)),mean_emb_i))
            if self.aggregate=="mean":
                intraLoss+=torch.mean(intraLoss_i)
            else:
                intraLoss+=torch.sum(intraLoss_i)
        mean_emb=torch.stack(mean_emb,dim=1) #(B,N2,k)

        # Calculate Inter Loss
        for i in range(1,M):
            for j in range(1,M):
                if (i!=j):
                    interLoss_ij=torch.mean(torch.cosine_similarity(mean_emb[:,i-1],mean_emb[:,j-1]))
                    interLoss+=interLoss_ij
        
        return intraLoss/M+interLoss/(M*(M-1))

def sdf_seg_loss(predSeg,trueSeg,segSDF):
    C=len(torch.unique(trueSeg))
    loss=0
    for i in range(C):
     loss+=torch.norm(predSeg[:,i,(trueSeg==1).nonzero()[:,1]]-segSDF[:,(trueSeg==1).nonzero()[:,1]],1)
    return loss        


class TotalSDFLoss(nn.Module):
    def __init__(self,seg_loss_type="ce",delta=0.01,lamda1=1,lamda2=10):
        super(TotalSDFLoss, self).__init__()
        self.seg_loss_type = seg_loss_type
        self.delta = delta # Lowering the delta value will put more weights on boundary learning
        self.lamda1 = lamda1
        self.lamda2 = lamda2



    def forward(self,predSeg,trueSeg,predSDF,trueSDF):
        ratio=trueSeg.unique(return_counts=True)[1]/torch.numel(trueSeg)
        #weights=1/ratio[predSDF.long()]
        self.loss1=DiceLoss(predSeg,trueSeg,1e-7,weights=1)
        self.loss2=AAAI_sdf_loss(predSDF,trueSDF,self.delta)

        loss=self.loss1+self.lamda2*self.loss2
        return loss



# Loss for Signed Distance Function
def AAAI_sdf_loss(net_output, gt_sdm,delta):
    """ From https://github.com/JunMa11/SegWithDistMap/blob/master/code/train_LA_AAAISDF.py"""
    smooth = 1e-5
    # Clamp the values between [-delta,delta]
    net_output=torch.clamp(net_output,min=-delta,max=delta)
    gt_sdm=torch.clamp(gt_sdm,min=-delta,max=delta)
    # Calculate the product loss
    intersect = net_output * gt_sdm
    pd_sum = net_output ** 2
    gt_sum = gt_sdm ** 2
    L_product = torch.sum((intersect + smooth) / (intersect + pd_sum + gt_sum + smooth))
    # Calculate the total loss
    L_SDF_AAAI = -L_product+torch.norm(net_output - gt_sdm, 1)

    return L_SDF_AAAI

def calculate_weight(ratio,alpha1=0.3,alpha2=0.3,alpha3=0.4):
    weight= 0.3+0.3*(1/(ratio+0.02))+0.4*(1/ratio)
    weight[weight==np.inf]==1
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
        w=w.expand(M,M)
        for i in range(M):
            pos_i=(true==i).nonzero()[:,1]
            pred_i=pred[:,:,pos_i]
            pred_label_i=pred_i.argmax(dim=1) # The prediction Label
            weight_i=w[i]
            weight_i=w[:,pred_label_i[0]]
            w_sum+=torch.sum(weight_i)

            loss+=torch.sum(-1*weight_i*torch.log(pred_i[:,i,:]))
    
        return loss/w_sum
