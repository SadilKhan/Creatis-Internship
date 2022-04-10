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
from imblearn.over_sampling import SMOTE,ADASYN


class PointCloudDataset(Dataset):
    """ Custom Class for Point Cloud Dataset """

    def __init__(self,csvDir,transforms=True,mode="train",expr_type="wm"):
        self.csvDir = csvDir+"/"+mode
        self.csvFiles=os.listdir(self.csvDir)
        self.transforms=transforms
        self.mode=mode
        self.expr_type = expr_type

        self.data=dict()
        for i in range(len(self.csvFiles)):
            pointSet,label=self.getValues(i)
            self.data[i]=[pointSet,label]


    def __len__(self):
        return len(self.csvFiles)
    
    def getValues(self,index):
        
        csvData=pd.read_csv(self.csvDir+"/"+self.csvFiles[index])
        pointSet=csvData[["x","y","z"]]
        label=csvData['label']

        if self.expr_type=="smote":
            sm=SMOTE()
            if self.mode=="train":
                pointSet,label=sm.fit_resample(pointSet,label)
        elif self.expr_type=="adasyn":
            ada=ADASYN()
            if self.mode=="train":
                pointSet,label=ada.fit_resample(pointSet,label)
        pointSet=pointSet.values

        # Normalize to Unit Ball
        pointSet-=np.mean(pointSet[:, :3], axis=0)
        pointSet/=np.max(np.sqrt(np.sum(pointSet ** 2, axis=1)), 0)
        
        # Labelling
        label=torch.tensor([ORGAN_TO_LABEL[l] for l in label])
        if self.transforms:
            pointSet=random_scale(pointSet)
            pointSet=translate_pointcloud(pointSet)
        else:
            pointSet=pointSet.astype(np.float32)
            
        pointSet=torch.from_numpy(pointSet)
        return pointSet,label        

    def __getitem__(self, index):
        pointSet,label=self.data[index]

        return pointSet,label        
