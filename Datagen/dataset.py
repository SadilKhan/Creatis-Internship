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

    def __int__(self,csvDir,transforms=False):
        self.csvDir = csvDir
        self.csvFiles=os.listdir(self.csvDir)
        self.transforms=transforms

    def __len__(self):
        return len(self.csvFiles)

    def __getitem__(self, index):
        csvData=pd.read(self.csvDir+"/"+self.csvFiles[0])

        pointSet=torch.from_numpy(csvData[["x","y","z","value","magnitude"]].values)

        # Normalize to Unit Ball
        pointSet[:,:3]-=torch.mean(pointSet[:, :3], axis=0)
        pointSet[:,:3]/=torch.max(np.sqrt(torch.sum(pointSet[:, :3] ** 2, axis=1)), 0)

        # Scale the remaining features
        pointSet[:,3:]-=torch.mean(pointSet[:, 3:], axis=0)
        pointSet[:,3:]/=torch.var(pointSet[:,3:],axis=0)    

        # Labelling
        label=torch.tensor([ORGAN_TO_LABEL[l] for l in csvData["label"]])

        if self.transforms:
            #pointSet[:,3:]=random_scale(pointSet[:,])
            #pointSet[:,3:]=translate_pointcloud(pointSet[:,])
            pass

        return pointSet,label            
