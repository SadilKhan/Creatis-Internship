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

def find_cube(segImage):
    """ Finds Bounding Cube of an segmentation image """

    x,y,z=np.where(segImage>0)
    A1=np.min(x),np.min(y),np.min(z)
    A2=np.max(x),np.max(y),np.max(z)

    
    return [A1,A2]


def sample_points(point1,point2,num=4000):
    """ Uniform Point Sampling """

    points=np.random.uniform(low=point1,high=point2,size=(num,3))
    points=np.round(points)
    points=points.astype(int)

    return points

def embed_points(points,imgData,segImage,gradx,grady,gradz,mag,prob=0.0,label=1):
    """ Filter and embed points which are in segmentation mask with features"""

    filteredPoints=[]

    for x,y,z in points:
        if segImage[x,y,z]>0:
            filteredPoints.append([x,y,z,imgData[x,y,z],gradx[x,y,z],grady[x,y,z],
            gradz[x,y,z],mag[x,y,z],label])
        # Add some of the background points with some probability
        elif np.random.random()<=prob:
            filteredPoints.append([x,y,z,imgData[x,y,z],gradx[x,y,z],grady[x,y,z],
            gradz[x,y,z],mag[x,y,z],"background"])

    
    return np.array(filteredPoints)
