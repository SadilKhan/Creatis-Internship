import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset,DataLoader
import platform
from utils import *

radLexIDDict={1247:"trachea",1302:"right lung",1326:"left lung",170:" pancreas",
187:"gallbladder",237:"urinary bladder",2473: "sternum",29193:"firdt lumber vertebra",
29662:"right kidney",29663:"left kidney",30324:"right adrenal gland",
30325:"left adrenal gland",32248:"right psoas major",32249:"left psoas major",
40357:"right rectus abdominis",40358: "left rectus abdominis",480:"aorta",
58: "liver",7578:"thyroid gland",86:"spleen"}

ORGAN_CHOICE={"liver":1,"pancreas":2,"spleen":3,"right lung":4,"left lung":5,"left kidney":6,"right kidney":7}

class PCDataset(Dataset):
    def __init__(self):
        pass



class VoxelData:

    """ Dataset for containing information about 3d Image and the segmentation Mask"""

    def __init__(self,imageDir,segDir,radLexIDDict,organChoice):
        
        # Image Name
        self.imageDir=imageDir 

        # Directory where all the segmentation masks are present
        self.segDir=segDir

        self.radLexIDDict=radLexIDDict
        self.oc=organChoice


        self.info()

        
    def info(self):
        # Load Image
        if platform.system()=="Windows":
            self.imageName=self.imageDir.split("'\'")[-1]
        else:
            self.imageName=self.imageDir.split("/")[-1]
        self.image=nib.load(self.imageDir)
        self.imageData=self.image.get_fdata() # Numpy Array


        try:
            # Get all the segmentation Mask
            self.allSegMask=os.listdir(self.segDir)
        except:
            raise Exception("Wrong Directory. Provide the Directory where all the segmentation masks are present.")

        # Patient ID
        self.patId=self.imageName.split("_")[0]

        # Segmentation Masks for the patients
        self.segMaskPatId=find_segmentation_mask(self.patId)

        # Dictionary for organ positions
        self.organSegPos=dict()

        for i in range(len(self.segMaskPatId)):
            tempSegMask=self.segMaskPatId[i]
            organId=tempSegMask.split("_")[-2]
            count=0

            if organId in self.oc:
                label=self.radLexIDDict[organId]
            else:
                label="background"
        
            try:
                self.organSegPos[label]+=find_pos_organ(self.segDir+"/"+tempSegMask)
            except:
                self.organSegPos[label]=find_pos_organ(self.segDir+"/"+tempSegMask)


        




        

