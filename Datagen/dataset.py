import csv
from unicodedata import name
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd
import os
import sys
import torch
from torch.utils.data import Dataset,DataLoader
import platform
from utils import *
from globalVar import *
from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class PointCloudDataset(Dataset):
    """ Custom Class for Point Cloud Dataset """

    def __init__(self,files,transforms=True,mode="train",expr_type="wm"):
        self.files=files
        self.transforms=transforms
        self.mode=mode
        self.expr_type = expr_type
        self.re=RandomOverSampler()

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,index):
        
        if "csv" in self.files[index]:
            csvData=pd.read_csv(self.files[index])
        else:
            csvData=pd.read_pickle(self.files[index])
        csvData.loc[csvData['label']!="liver","label"]="background" # Change all the labels to background except lungs
        name =self.files[index].split("/")[-1].split("_")[0]
        csvData.loc[(csvData['label']!="background") & (csvData['SDF']>0),'SDF']=-1*csvData.loc[(csvData['label']!="background") & (csvData['SDF']>0),'SDF']
        label=csvData.pop("label")
        sdf=csvData.pop("SDF")

        try:
            pointSet=csvData[["x","y","z"]]
            features=np.vstack(csvData["neighbor_intensity"])
            features=np.clip(features,a_min=-250,a_max=250)
            n=int(np.ceil(features.shape[1]**(1/3)))
        except:
            pointSet=csvData[["x","y","z"]]
            features=None
        
        pointSet=pointSet.values
        sdf=sdf.values

        # Normalize to Unit Ball
        pointSet-=np.mean(pointSet, axis=0)
        pointSet/=np.max(np.sqrt(np.sum(pointSet ** 2, axis=1)),0)

        # Standard Scaling
        se=StandardScaler()
        features=se.fit_transform(features)
        sdf=sdf/np.abs(sdf).max()
        
        # Labelling
        label=torch.tensor([ORGAN_TO_LABEL[l] for l in label]) # ORGAN_TO_LABEL has some changes
        pointSet=pointSet.astype(np.float32)
        pointSet=torch.from_numpy(pointSet)
        features=features.astype(np.float32)
        features=features.reshape(-1,n,n,n)
        features=torch.from_numpy(features)
        sdf=sdf.astype(np.float32)
        sdf=torch.from_numpy(sdf)

        return name,pointSet,features,label,sdf
