import argparse
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import os
import matplotlib.pyplot as plt
from globalVar import *


def find_segmentation_mask(segDir,patNum):
    """ Finds all the segmentation masks according to Patient Id"""
    
    # Get all the segmentation names
    segMaskNames=np.array(os.listdir(segDir))

    # Split the segmentation mask strings
    allSegMaskSpl=[i.split("_") for i in segMaskNames]
    
    # The first item is the patient number
    allIndex=[i for i in range(len(allSegMaskSpl)) if allSegMaskSpl[i][0]==str(patNum)]
    return segMaskNames[allIndex]

def seg_to_class(segMask):
    """ Seg ID to Class Name"""
    Id=segMask.split("_")[-2]

    try:
        return radLexIDDict[Id]
    except:
        return "background"

def transform_to_ras(imageName):
    """ Function to transform Image orientation to RAS """

    image=nib.load(imageName)
    orientation=nib.aff2axcodes(image.affine)

    if orientation!=('R', 'A', 'S'):
        fimage=nib.as_closest_canonical(image)
        return fimage
    else:
        print("IMAGE ALREADY IN RAS+ FORMAT")
        return image



def find_corners(segImageSlice):
    """ Finds the bounding box of a 2d organ image"""
    x,y=(segImageSlice>0).nonzero()

    [x1,y1]=[x[0],np.min(y)]
    [x2,y2]=[x[-1],np.max(y)]

    return [x1,y1],[x2,y2]

def find_cube(segImage):
    """ Finds Bounding Cube of an object """
    startIndex=0
    stopIndex=0

    h,w,d=segImage.shape

    # Finds the position along z axis when the organ mask starts
    for i in range(d):
        if (segImage[:,:,i]>0).any():
            startIndex=i
            break 

    # Finds the position along z axis when the organ mask ends        
    for j in range(d-1,0,-1):
        if (segImage[:,:,j]>0).any():
            stopIndex=j
            break   
    
    cube=[[np.inf,np.inf,startIndex],[0,0,stopIndex]]
    
    for i in range(startIndex,stopIndex+1):
        A,B=find_corners(segImage[:,:,i])

        for j in range(2):
            if cube[0][0]>A[0]:
                cube[0][0]=A[0]
            if cube[0][1]>A[1]:
                cube[0][1]=A[1]
            if cube[1][0]<B[0]:
                cube[1][0]=B[0]
            if cube[1][1]<B[1]:
                cube[1][1]=B[1]
    
    return cube


def sample_points(point1,point2,num=4000):
    """ Uniform Point Sampling """

    points=np.random.uniform(low=point1,high=point2,size=(num,3))
    points=np.round(points)
    points=points.astype(int)

    return points

def filter_points(points,segImage,prob=0.0,label=1):
    """ Filter points which are in segmentation mask """

    filteredPoints=[]

    for x,y,z in points:
        if segImage[x,y,z]>0:
            filteredPoints.append([x,y,z,label])
        # Add some of the background points
        elif np.random.random()<=prob:
            filteredPoints.append([x,y,z,"background"])

    
    return np.array(filteredPoints)


def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return torch.sqrt(dists).float()

def fps(xyz, M):
    '''
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    dists = torch.ones(B, N).to(device) * 1e5
    inds = torch.randint(0, N, size=(B, ), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :] # (B, 3)
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz))
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    return centroids

