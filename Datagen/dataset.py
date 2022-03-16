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


class PointCloudDataset(Dataset):
    """ Custom Class for Point Cloud Dataset """

    def __init__(self,csvDir,transforms=True,mode="train",n_samples=100000):
        self.csvDir = csvDir +"/"+mode
        self.csvFiles=os.listdir(self.csvDir)
        self.transforms=transforms
        self.n_samples=n_samples

    def __len__(self):
        return len(self.csvFiles)

    def __getitem__(self, index):
        csvData=pd.read_csv(self.csvDir+"/"+self.csvFiles[0])
        allIndices=csvData[csvData['label']=="background"].index
        chosenIndices=np.random.choice(allIndices,50000)
        backData=csvData.loc[chosenIndices]
        csvData=csvData[csvData['label']!="background"]

        #csvData=pd.concat([csvData,backData],ignore_index=True)
        #indices=np.random.choice(csvData.index,100000)
        #csvData=csvData.loc[indices]

        pointSet=csvData[["x","y","z"]].values

        # Normalize to Unit Ball
        pointSet-=np.mean(pointSet[:, :3], axis=0)
        pointSet/=np.max(np.sqrt(np.sum(pointSet ** 2, axis=1)), 0)

        """# Scale the remaining features
        pointSet[:,3:]-=np.mean(pointSet[:, 3:], axis=0)
        pointSet[:,3:]/=np.var(pointSet[:,3:],axis=0)    """

        
        # Labelling
        label=torch.tensor([ORGAN_TO_LABEL[l] for l in csvData["label"]])

        if self.transforms:
            pointSet=random_scale(pointSet)
            pointSet=translate_pointcloud(pointSet)
        else:
            pointSet=pointSet.astype(np.float32)
            
        pointSet=torch.from_numpy(pointSet)

        return pointSet,label            
