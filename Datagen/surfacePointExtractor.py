import numpy as np
import argparse
import os,glob,sys
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import itertools
sys.path.append("/home/khan/Internship/Codes/Model/model_utils")
from knn import findKNN
from globalVar import *
import open3d as o3d


    
def extractSurfacePoints(imageData):
    """ Extracts the points of the organs"""
    # Load the Mask
    X,Y,Z=(imageData>0).nonzero() # Positions for the organs 
    boundaryPoints=[] # All the boundary points
    # Find all the points that have one background point in 3x3x3 cube centered around the point
    for i in range(len(X)):
        x,y,z=find_neighbor_points(X[i],Y[i],Z[i],3,imageData)
        if np.sum(imageData[x,y,z]>0)<3**3:
            boundaryPoints.append([X[i],Y[i],Z[i]])
    bdPointsDf=pd.DataFrame(boundaryPoints,columns=["x","y","z"])
    return bdPointsDf


def find_neighbor_points(x,y,z,nbr_fs=3,imageData=None):
    min_,max_=(1-nbr_fs)//2,(nbr_fs-1)//2
    x0=np.array([x+i for i in range(min_,max_+1)],dtype=np.int32).clip(0,imageData.shape[0]-1)
    y0=np.array([y+i for i in range(min_,max_+1)],dtype=np.int32).clip(0,imageData.shape[1]-1)
    z0=np.array([z+i for i in range(min_,max_+1)],dtype=np.int32).clip(0,imageData.shape[2]-1)

    pos=[[],[],[]]
    for i in itertools.product(z0,x0,y0):
        pos[2].append(i[0])
        pos[0].append(i[1])
        pos[1].append(i[2])
    return pos

def sdfCalculator(bdPoints,allPoints):
    """ Calculate the Signed Distance Function """
    idx,dist=findKNN(bdPoints.values.reshape(1,-1,3).astype(np.float32),allPoints[['x','y','z']].values.reshape(1,-1,3).astype(np.float32),1)
    allPoints['SDF']=dist.flatten().numpy()

    return allPoints


if __name__ == "__main__":
    pass