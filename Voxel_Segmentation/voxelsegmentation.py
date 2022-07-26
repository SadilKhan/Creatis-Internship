
import os,gc
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

class VoxelSegmentation():
    def __init__(self,modelPath,csvPath,imagePath,segPath1,segPath2,outputPath,organ):
        self.modelPath = modelPath
        self.csvPath = csvPath
        self.imagePath = imagePath
        self.segPath1 = segPath1
        self.segPath2 = segPath2
        self.outputPath = outputPath
        self.organ=organ
        self.segment()
    
    def preprocess(self):
        csvData=pd.read_pickle(self.csvPath)
        csvData.loc[csvData['label']!=self.organ,"label"]="background" # Change all the labelsPar to background except lungs
        name = self.csvPath.split("/")[-1].split("_")[0]
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
        #scores,_,_,emb,_=self.model(points.unsqueeze(0).cuda(),nbr_intensity.unsqueeze(0).cuda())
        #predLabel=scores.argmax(dim=1)[0].cpu()
        #pos=(predLabel==1).nonzero().flatten()
        #self.minpoint=opoints[:,pos].min(dim=1).values[0]
        #self.maxpoint=opoints[:,pos].max(dim=1).values[0]
        #print(self.minpoint,self.maxpoint)
        self.minpoint=torch.tensor([0., 0., 0.])
        self.maxpoint=torch.tensor([127., 127.,  111.])
        self.shape=[int(i) for i in self.maxpoint-self.minpoint]

        self.labelPred=[]
        self.probPred=[]
        onewPoints=self.getAllVoxels(self.minpoint,self.maxpoint)
        self.allIntensities=self.imageData[onewPoints[:,0],onewPoints[:,1],onewPoints[:,2]].reshape(self.shape[0],self.shape[1],self.shape[2])
        newpoints=((onewPoints.double()@self.rotMat+self.transMat)-mean)/norm
        exnewpoints=newpoints
        self.allLabels=self.findAllLabels(self.segData,onewPoints)
        #print(self.allLabels.shape)
        cm=np.array([[0,0],[0,0]])
        p=onewPoints.shape[0]//100000
        self.actualLabels=[]
        for i in tqdm(range(p+1)):
            new_nbr_intensity=[find_neighbor_intensity(onewPoints[i,0].item(),onewPoints[i,1].item(),onewPoints[i,2].item(),7,self.imageData) for i in range(np.clip(100000*(i),a_min=0,a_max=onewPoints.shape[0]),np.clip(100000*(i+1),a_min=0,a_max=onewPoints.shape[0]))]
            new_nbr_intensity=np.stack(new_nbr_intensity)
            new_nbr_intensity=np.clip(new_nbr_intensity,a_min=-250,a_max=250)
            se=StandardScaler()
            new_nbr_intensity=se.fit_transform(new_nbr_intensity)
            new_nbr_intensity=torch.from_numpy(new_nbr_intensity).reshape((-1,7,7,7))
            # Stack points and newpoints
            newpoints=torch.concat([points.unsqueeze(0),exnewpoints[100000*i:100000*(i+1)].unsqueeze(0)],dim=1)
            
            new_nbr_intensity=torch.concat([nbr_intensity.unsqueeze(0),new_nbr_intensity.unsqueeze(0)],dim=1)
            gc.collect()
            torch.cuda.empty_cache()
            scores,sdf,outScores,_,_,_=self.model(newpoints.float().cuda(),new_nbr_intensity.float().cuda())
            #print(points.shape)
            predLabel=(sdf<=0)*1
            #predLabel=outScores.argmax(dim=1)
            self.probPred.append(sdf[:,points.shape[0]:].detach().cpu())
            del sdf
            #predLabel=(scores>0.5)*1
            cm+=confusion_matrix(self.allLabels[100000*i:100000*(i+1)],predLabel[0,points.shape[0]:].detach().cpu())
            posNew=(predLabel[:,points.shape[1]:]==1).nonzero()[:,1]
            #self.actualLabels.append(self.allLabels[100000*i:100000*(i+1)][posNew.cpu()])
            self.labelPred.append(predLabel[:,points.shape[0]:].detach().cpu())
        self.probPred=np.concatenate(self.probPred,axis=1)[0].reshape(self.shape[0],self.shape[1],self.shape[2])
        self.labelPred=np.concatenate(self.labelPred,axis=1)[0].reshape(self.shape[0],self.shape[1],self.shape[2])
        #self.labelPred=torch.cat(self.labelPred,axis=1)[0]
        #print(self.labelPred.shape)
        self.allLabels=self.allLabels.reshape(self.shape[0],self.shape[1],self.shape[2])
        print(cm)

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
        nib.save(nib.Nifti1Image(self.labelPred.astype(np.int16),self.image.affine),self.outputPath+"/"+"predLabel.nii.gz")
        nib.save(nib.Nifti1Image(self.allLabels.astype(np.int16),self.image.affine),self.outputPath+"/"+"actualLabel.nii.gz")
        nib.save(nib.Nifti1Image(self.probPred.astype(np.float32),self.image.affine),self.outputPath+"/"+"probPred.nii.gz")
    

def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--modelPath",default="/home/khan/Internship/Codes/model_lungs_w20_two_head_sdf12.pth")
    parser.add_argument("--csvPath",help="CSV Path for point cloud",default="/home/khan/Internship/dataset/oth/pc/10000112_1_CTce_ThAb_edge_20_50_point_cloud.pkl")
    parser.add_argument("--imagePath",help="Image Path",default="/home/khan/Internship/dataset/visceral/volumes/CTce_ThAb/10000112_1_CTce_ThAb.nii.gz")
    parser.add_argument("--segPath1",help="Segmentation Path for point cloud",default="/home/khan/Internship/dataset/visceral/segmentations/10000112_1_CTce_ThAb_1326_8.nii.gz")
    parser.add_argument("--segPath2",help="Segmentation Path for point cloud",default="/home/khan/Internship/dataset/visceral/segmentations/10000112_1_CTce_ThAb_1302_8.nii.gz")
    parser.add_argument("--outputPath",help="Output Path for point cloud",default="/home/khan/Internship/")
    parser.add_argument("--organ",default="lungs")

    args=parser.parse_args()

    VoxelSegmentation(args.modelPath,args.csvPath,args.imagePath,args.segPath1,args.segPath2,args.outputPath,args.organ)

if __name__=="__main__":
    main()

    
    
