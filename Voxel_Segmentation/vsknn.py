import os,gc
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
CURRRENT_DIRECTORY =os.getcwd()
BASE_DIRECTORY="/".join(CURRRENT_DIRECTORY.split("/")[:-1])
sys.path.append(BASE_DIRECTORY)
sys.path.append(BASE_DIRECTORY+"/Datagen")
sys.path.append(BASE_DIRECTORY+"/Model")

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from Datagen.dataset import PointCloudDataset
from torchsummary import summary
import pynanoflann
from model import RandLANet
from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler
from Datagen.utils import *
from sklearn.metrics import *
from Datagen.utils import *
import torch.nn.functional as F
from model_utils.metrics import *
from model_utils.dice_loss import *
from model_utils.tools import Config as cfg
from model_utils.tools import *
from sklearn.preprocessing import StandardScaler
from model_utils.knn import findKNN
from utils import *

class VoxelSegmentation:
    def __init__(self,modelPath,csvPath,imagePath,segPath1,segPath2,neighbor,outputPath,organ):
        self.modelPath = modelPath
        self.csvPath = csvPath
        self.imagePath = imagePath
        self.segPath1 = segPath1
        self.segPath2 = segPath2
        self.neighbor=neighbor
        self.outputPath=outputPath
        self.organ=organ
        
        self.segment()
    
    def preprocess(self):
        csvData=pd.read_pickle(self.csvPath)
        csvData.loc[csvData['label']!=self.organ,"label"]="background" # Change all the labelsPar to background except lungs
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

        opoints=csvData[['xo','yo','zo']].values
        opoints=torch.from_numpy(opoints).unsqueeze(0)

        return name,pointSet,opoints,features,mean,norm,label
    
    def load(self):
        # Image
        self.image = nib.load(self.imagePath)
        self.imageData = self.image.get_fdata()
        self.rotMat=torch.from_numpy(self.image.affine[:3, :3])
        self.invrotMat = torch.inverse(self.rotMat)
        self.transMat = torch.from_numpy(self.image.affine[:3, 3])
        # Segmentation Label
        self.seg1=transform_to_ras(self.segPath1)
        self.seg1Data=self.seg1.get_fdata()
        if self.segPath2!="No":
            self.seg2=transform_to_ras(self.segPath2)
            self.seg2Data=self.seg2.get_fdata()
            self.segData=self.seg1Data+self.seg2Data
        else:
            self.segData=self.seg1Data
        # Model
        self.model=RandLANet(3,2,8,2,torch.device('cuda')).cuda()
        self.model.load_state_dict(torch.load(self.modelPath))
    
    def segment(self):
        name,points,opoints,nbr_intensity,mean,norm,labels=self.preprocess()
        self.load()
        
        # Find the scores for all the input point cloud
        _,sdf,_,_,_=self.model(points.unsqueeze(0).cuda(),nbr_intensity.unsqueeze(0).cuda())

        # Create a grid box
        self.minpoint=torch.tensor([0,0,0])
        self.maxpoint=torch.tensor(self.imageData.shape)
        self.shape=[int(i) for i in self.maxpoint-self.minpoint]

        # Get the new points
        onewPoints=self.getAllVoxels(self.minpoint,self.maxpoint)
        self.allIntensities=self.imageData[onewPoints[:,0],onewPoints[:,1],onewPoints[:,2]].reshape(self.shape[0],self.shape[1],self.shape[2])
        newpoints=((onewPoints.double()@self.rotMat+self.transMat)-mean)/norm
        #print(points.shape)

        # Calculate weights for interpolation
        idx,dist=findKNN(points.unsqueeze(0).cpu(),newpoints.unsqueeze(0),num_neighbors=self.neighbor) # dist shape (B,N,k)
        sdf_neighbor=sdf[:,idx[0]].detach().cpu() # Shape (B,N,k)
        distInv=1/dist
        posInf=(distInv==np.inf).nonzero()[:,1]
        # Replace the weights with 1 for the duplicate points. For duplicate point, weight=[1,0,0,...,0]
        value=torch.zeros(distInv[:,posInf].shape)
        value[:,:,0]+=1

        distInv[:,posInf]=value
        #print(distInv.shape)
        weights=distInv/distInv.sum(axis=-1).unsqueeze(-1)

        # Interpolate the sdf
        sdf=(sdf_neighbor*weights).sum(-1)
        self.predLabel=(sdf[0]<=0)*1
        self.allLabels=self.findAllLabels(self.segData,onewPoints)

        cm=confusion_matrix(self.allLabels,self.predLabel)
        print(cm)

        self.predLabel=np.array(self.predLabel).reshape(self.shape[0],self.shape[1],self.shape[2])
        self.gtLabel=np.array(self.allLabels).reshape(self.shape[0],self.shape[1],self.shape[2])
        self.sdfPred=np.array(sdf.detach().cpu()).reshape(self.shape[0],self.shape[1],self.shape[2])
        #print(self.predLabel.astype(np.float32))

        self.save()

    def getAllVoxels(self,minPoint,maxPoint):
        allPoints=[]
        gx,gy,gz=maxPoint-minPoint
        allPointsX=torch.arange(minPoint[0],maxPoint[0]).reshape(-1,1)
        allPointsY=torch.arange(minPoint[1],maxPoint[1]).reshape(-1,1)
        allPointsZ=torch.arange(minPoint[2],maxPoint[2]).reshape(-1,1)
        for i in itertools.product(allPointsX,allPointsY,allPointsZ):
            allPoints.append([i[0],i[1],i[2]])
        allPoints=torch.tensor(allPoints)
        allPoints=allPoints.long()
        return allPoints
    def findAllLabels(self,segData,pt,multiplier=4):
        pt=pt*4
        X=pt[:,0]
        Y=pt[:,1]
        Z=pt[:,2]

        label=(segData[X,Y,Z]>0)*1
        return label
    
    def save(self):
        nib.save(nib.Nifti1Image(self.allIntensities,self.image.affine),self.outputPath+"/"+"intensity.nii.gz")
        nib.save(nib.Nifti1Image(self.predLabel.astype(np.float32),self.image.affine),self.outputPath+"/"+"predLabel.nii.gz")
        nib.save(nib.Nifti1Image(self.gtLabel.astype(np.float32),self.image.affine),self.outputPath+"/"+"actualLabel.nii.gz")
        nib.save(nib.Nifti1Image(self.sdfPred.astype(np.float32),self.image.affine),self.outputPath+"/"+"sdfPred.nii.gz")

    

def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--modelPath",default="/home/khan/Internship/Codes/model_sdf2.pth")
    parser.add_argument("--csvPath",help="CSV Path for point cloud",default="/home/khan/Internship/dataset/oth/pc/10000112_1_CTce_ThAb_edge_20_50_point_cloud.pkl")
    parser.add_argument("--imagePath",help="Image Path",default="/home/khan/Internship/dataset/visceral/volumes/CTce_ThAb/10000112_1_CTce_ThAb.nii.gz")
    parser.add_argument("--segPath1",help="Segmentation Path for point cloud",default="/home/khan/Internship/dataset/visceral/segmentations/10000112_1_CTce_ThAb_1326_8.nii.gz")
    parser.add_argument("--segPath2",help="Segmentation Path for point cloud",default="/home/khan/Internship/dataset/visceral/segmentations/10000112_1_CTce_ThAb_1302_8.nii.gz")
    parser.add_argument("--neighbor",help="KNN neighbor",type=int,default=8)
    parser.add_argument("--outputPath",help="Output Path for point cloud",default="/home/khan/Internship/")
    parser.add_argument("--organ",help="Organ",default="lungs")

    args=parser.parse_args()

    vs=VoxelSegmentation(args.modelPath,args.csvPath,args.imagePath,args.segPath1,args.segPath2,args.neighbor,args.outputPath,args.organ)


    
    
if __name__== "__main__": 
    main()
