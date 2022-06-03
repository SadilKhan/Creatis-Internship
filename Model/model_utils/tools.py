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

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

class Config:
    sampling_type = 'active_learning'
    #class_weights_visceral=[207194.95, 5628.1, 23958, 860.1, 1123.25, 1048.4]
    class_weights_visceral=[207194.95, 23958]
    class_weights_ctorg=[213092,5554,16651,610,1408,1419]

class DataProcessing:
    @staticmethod
    def get_file_list(dataset_path, test_scan_num):
        seq_list = np.sort(os.listdir(dataset_path))

        train_file_list = []
        test_file_list = []
        val_file_list = []
        for seq_id in seq_list:
            seq_path = join(dataset_path, seq_id)
            pc_path = join(seq_path, 'velodyne')
            if seq_id == '08':
                val_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
                if seq_id == test_scan_num:
                    test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif int(seq_id) >= 11 and seq_id == test_scan_num:
                test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                train_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        train_file_list = np.concatenate(train_file_list, axis=0)
        val_file_list = np.concatenate(val_file_list, axis=0)
        test_file_list = np.concatenate(test_file_list, axis=0)
        return train_file_list, val_file_list, test_file_list

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending of the input
        """

        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU



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
        super().__init__()
        self.alpha=0.7
        self.beta=1.5
        self.weight=weight
        self.aggregate=aggregate


    def forward(self,points,true,embedding):
        """
        points: shape (B,N,3)
        true: shape (B,N,C)
        embedding: shape (B,N,k)
        
        """
        M=len(torch.unique(true)) # Number of Labels
        N=points.size(1) # Number of points
        intraLoss=0
        interLoss=0
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
            """            if i==0:
                weight_i=w[:,pred_label_i[0]]
                w_sum+=torch.sum(weight_i*freq[:,i])
            else:"""
            weight_i=w[i]
            weight_i=w[:,pred_label_i[0]]
            w_sum+=torch.sum(weight_i)

            loss+=torch.sum(-1*weight_i*torch.log(pred_i[:,i,:]))
    
        return loss/w_sum

class FPCELossV2(nn.Module):
    """ Cross Entopy Loss which takes into account the all the false positive error"""
    def __init__(self):
        super(FPCELossV2, self).__init__()
        pass
    def forward(self,pred,true,weight="bfn"):
        """
        pred: shape (B,C,N)
        true: shape (B,N)
        """
        pred=F.softmax(pred,dim=1)
        M=pred.size(1)
        loss=0
        if weight=="cm":
            # Calculate Weights
            pred_label=pred.argmax(dim=1)
            cm=confusion_matrix(true[0].cpu(),pred_label[0].cpu())
            cm+=1
            cmRatio=cm/cm.sum()
            w=calculate_weight(cmRatio) # Updates weights for error cell
            #np.fill_diagonal(w,np.diagonal(calculate_weight(1-cmRatio))) # Updates the true positive cell
            w=torch.tensor(w)
            w=w.cuda()
            #w+=0.01
            w_sum=0
            for i in range(M):
                pos_i=(true==i).nonzero()[:,1]
                pred_i=pred[:,:,pos_i]
                pred_label_i=pred_i.argmax(dim=1) # The prediction Label
                weight_i=w[i,pred_label_i[0]]
                w_sum+=torch.sum(weight_i)
                loss+=torch.sum(-1*weight_i*torch.log(pred_i[:,i,:]))
        else:
            freq=torch.einsum("bch->bh", F.one_hot(true)).type(torch.float32)
            ratio=freq/torch.sum(freq)
            w=calculate_weight(ratio)
            w_sum=0
            for i in range(M):
                pos_i=(true==i).nonzero()[:,1]
                pred_i=pred[:,:,pos_i]
                pred_label_i=pred_i.argmax(dim=1) # The prediction Label
                if i==0:
                    weight_i=w[:,pred_label_i[0]]
                    w_sum+=torch.sum(weight_i)
                else:
                    weight_i=w[:,i]
                    w_sum+=torch.sum(weight_i*freq[:,i])

                loss+=torch.sum(-1*weight_i*torch.log(pred_i[:,i,:]))
    
        return loss/w_sum

    
class FPCELossV3(nn.Module):
    """ Cross Entopy Loss which takes into account the all the false positive error"""
    def __init__(self):
        super(FPCELossV3, self).__init__()
        pass
    def forward(self,pred,true):
        """
        pred: shape (B,C,N)
        true: shape (B,N)
        """
        pred=F.softmax(pred,dim=1)
        M=pred.size(1)
        N=true.size(1)
        loss=0
        for i in range(M):
            pos_i=(true==i).nonzero()[:,1]
            pred_i=pred[:,:,pos_i]
            loss+=torch.sum((-1/pred_i[:,i,:])*torch.log(pred_i[:,i,:]))
        #print(pred_i)
        return loss/N

    