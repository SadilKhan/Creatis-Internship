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

def embed_points(points,imgData,segImage,mag,prob=0.0,label="background"):
    """ Filter and embed points which are in segmentation mask with features"""

    filteredPoints=[]

    for x,y,z in points:
        if segImage[x,y,z]>0:
            filteredPoints.append([x,y,z,imgData[x,y,z],mag[x,y,z],label])
        # Add some of the background points with some probability
        elif np.random.random()<=prob:
            filteredPoints.append([x,y,z,imgData[x,y,z],mag[x,y,z],"background"])

    
    return np.array(filteredPoints)



def random_scale(point_data, scale_low=0.8, scale_high=1.2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original batch of point clouds
        Return:
            Nx3 array, scaled batch of point clouds
    """
    scale = np.random.uniform(low=scale_low, high=scale_high, size=[3])
    scaled_pointcloud = np.multiply(point_data, scale).astype('float32')
    return scaled_pointcloud


def translate_pointcloud(pointcloud):
    shift = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(pointcloud, shift).astype('float32')
    return translated_pointcloud