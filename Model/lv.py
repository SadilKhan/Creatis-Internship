import os,gc
import pickle
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
sys.path.append("/home/khan/Internship/Codes/Model/")
sys.path.append("/home/khan/Internship/Codes/Datagen/")

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torchsummary import summary
import pynanoflann
from model_original import RandLANet
from sklearn.metrics import *
import torch.nn.functional as F
from model_utils.metrics import *
from model_utils.dice_loss import *
from model_utils.tools import Config as cfg
from model_utils.tools import *
from sklearn.preprocessing import StandardScaler
from model_utils.knn import findKNN
from utils import *


class LatentVector():
    def __init__(self,args):
        self.modelPath = args.modelPath
        self.csvPath = args.csvPath
        self.outputPath = args.outputPath
        self.organ=args.organ

        self.diff_latent_vector()
    
    def load(self):
        # Model
        self.model=RandLANet(3,6,8,2,torch.device('cuda')).cuda()
        self.model.load_state_dict(torch.load(self.modelPath))


    def preprocess(self):
        csvData=pd.read_pickle(self.csvPath)
        #csvData.loc[csvData['label']!=self.organ,"label"]="background" # Change all the labelsPar to background except lungs
        name =self.csvPath.split("/")[-1].split("_")[0]
        label=csvData.pop("label")

        try:
            pointSet=csvData[["x","y","z"]]
            features=np.vstack(csvData["neighbor_intensity"])
            features=np.clip(features,a_min=-250,a_max=250)
            n=int(np.ceil(features.shape[1]**(1/3)))
        except:
            pointSet=csvData[["x","y","z"]]
            features=None
        
        pointSet=pointSet.values

        # Normalize to Unit Ball
        mean=np.mean(pointSet[:, :3], axis=0)
        pointSet[:, :3]-=mean
        norm=np.max(np.sqrt(np.sum(pointSet[:, :3]** 2, axis=1)),0)
        pointSet[:, :3]/=norm

        # Standard Scaling
        se=StandardScaler()
        features=se.fit_transform(features)
        
        # Labelling
        label=torch.tensor([ORGAN_TO_LABEL[l] for l in label]) # ORGAN_TO_LABEL has some changes
        pointSet=pointSet.astype(np.float32)
        pointSet=torch.from_numpy(pointSet)
        features=features.astype(np.float32)
        features=features.reshape(-1,n,n,n)
        features=torch.from_numpy(features)

        return name,pointSet,features,label

    def diff_latent_vector(self):
        name,pointSet,features,label=self.preprocess()
        self.load()
        scores,lv,permutation=self.model(pointSet.unsqueeze(0).cuda(),features.unsqueeze(0).cuda())
        self.lvDiff=dict()
        label=label[permutation]
        label=label[:lv.shape[2]]

        # Calculate Mean Embedding vector for input organ
        self.organID=ORGAN_TO_LABEL[self.organ]
        pos=(label==self.organID).nonzero().flatten()
        meanEmb=torch.mean(lv[:,:,pos])

        numLabel=torch.unique(label).shape[0]

        for i in tqdm(range(numLabel)):
            pos=(label==i).nonzero().flatten()
            emb=lv[:,:,pos]
            diff=torch.sum((emb-meanEmb)**2,axis=1)
            self.lvDiff[LABEL_TO_ORGAN[i]]=diff
       

        file=open(self.outputPath+"/"+"lv.pickle","wb")
        pickle.dump(self.lvDiff,file)
        pickle.dump(permutation,open(self.outputPath+"/permutation.pickle","wb"))
        pickle.dump(label,open(self.outputPath+"/label.pickle","wb"))
        pickle.dump(scores.detach().cpu(),open(self.outputPath+"/pred.pickle","wb"))
        pickle.dump(pointSet,open(self.outputPath+"/points.pickle","wb"))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--modelPath",default="/home/khan/Internship/Codes/BEST_MODEL_V1.pth")
    parser.add_argument("--csvPath",default="/home/khan/Internship/dataset/Extracted/point_cloud/visceral_7/10000112_1_CTce_ThAb_edge_5.0_50.0_point_cloud.pkl")
    parser.add_argument("--organ",default="lungs")
    parser.add_argument("--outputPath",default="/home/khan/Internship")

    args = parser.parse_args()

    lv=LatentVector(args)

if __name__ == "__main__":
    main()



        
        

