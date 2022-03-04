from re import S
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset,DataLoader
import platform
from utils import *
from globalVar import *


class PointCloudDataset(Dataset):
    """ Custom Class for Point Cloud Dataset """

    def __init__(self,csvDir,transforms=True,mode="train"):
        self.csvDir = csvDir +"/"+mode
        self.csvFiles=os.listdir(self.csvDir)
        self.transforms=transforms

    def __len__(self):
        return len(self.csvFiles)

    def __getitem__(self, index):
        csvData=pd.read_csv(self.csvDir+"/"+self.csvFiles[0])

        pointSet=csvData[["x","y","z","value","magnitude"]].values

        # Normalize to Unit Ball
        pointSet[:,:3]-=np.mean(pointSet[:, :3], axis=0)
        pointSet[:,:3]/=np.max(np.sqrt(np.sum(pointSet[:, :3] ** 2, axis=1)), 0)

        # Scale the remaining features
        pointSet[:,3:]-=np.mean(pointSet[:, 3:], axis=0)
        pointSet[:,3:]/=np.var(pointSet[:,3:],axis=0)    

        # Labelling
        label=torch.tensor([ORGAN_TO_LABEL[l] for l in csvData["label"]])

        if self.transforms:
            pointSet[:,:3]=random_scale(pointSet[:,:3])
            pointSet[:,:3]=translate_pointcloud(pointSet[:,:3])
            
        pointSet=torch.from_numpy(pointSet)
        return pointSet[:,:3],label            
