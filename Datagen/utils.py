import argparse
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import os
import matplotlib.pyplot as plt
from globalVar import *
from scipy import ndimage

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
        #print("IMAGE ALREADY IN RAS+ FORMAT")
        return image

def find_cube(segImage,label):
    """ Finds Bounding Cube of an segmentation image """

    x,y,z=np.where(segImage==label)
    A1=[np.min(x),np.min(y),np.min(z)]
    A2=[np.max(x),np.max(y),np.max(z)]

    
    return np.array([A1,A2])


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


def dilation(image):

    # Dialte the image
    dilated=ndimage.binary_dilation(image,iterations=1)

    return dilated


def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N 
    """
    points = np.array(points)
    
    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected 
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]
        
        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, 
                                        dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return sample_inds

def ctorg_find_seg_mask(dir,imgNum=-1):
    """ Finds the segmentation masks for CT ORG datasets"""
    allFile=os.listdir(dir)
    segMasks=[]

    if imgNum<0:
        for file in allFile:
            if "labels" in file:
                segMasks.append(file)
        return segMasks
    else:
        file=f"labels-{imgNum}.nii.gz"
        return file
def ctorg_find_volume_mask(dir,imgNum=-1):
    """ Finds the volumes masks for CT ORG datasets"""

    allFile=os.listdir(dir)
    volMasks=[]

    if imgNum<0:
        for file in allFile:
            if "volume" in file:
                volMasks.append(file)
        return volMasks
    else:
        file=f"volume-{imgNum}.nii.gz"
        return file

  
def getInfo(data):
    n=len(data)
    trainLoss=[]
    testLoss=[]
    trAcc,tstAcc=[],[]
    trIou,tstIou=[],[]
    trainAcc,trainIou,testAcc,testIou=dict(),dict(),dict(),dict()
    for i in range(1,n):
        if "Training loss:" in data[i]:
            trainLoss.append(float(data[i].split(" ")[2].split("\t")[0]))
            testLoss.append(float(data[i].split(" ")[-1].split("\t")[0]))
        if ("Training" in data[i]) and ("Accuracy" in data[i-1]):
            for j,d in enumerate(data[i].split("|")[1:-1]):
                try:
                    trainAcc[LABEL_TO_ORGAN[j]].append(float(d.split("/n")[0]))
                except:
                    trainAcc[LABEL_TO_ORGAN[j]]=[float(d.split("/n")[0])]
            trAcc.append(float(float(data[i].split("|")[-1].split("/n")[0])))

        if ("Validation" in data[i]) and ("Accuracy" in data[i-2]):
            for j,d in enumerate(data[i].split("|")[1:-1]):
                try:
                    testAcc[LABEL_TO_ORGAN[j]].append(float(d.split("/n")[0]))
                except:
                    testAcc[LABEL_TO_ORGAN[j]]=[float(d.split("/n")[0])]
            tstAcc.append(float(float(data[i].split("|")[-1].split("/n")[0])))
        
        if ("Training" in data[i]) and ("IoU" in data[i-1]):
            for j,d in enumerate(data[i].split("|")[1:-1]):
                try:
                    trainIou[LABEL_TO_ORGAN[j]].append(float(d.split("/n")[0]))
                except:
                    trainIou[LABEL_TO_ORGAN[j]]=[float(d.split("/n")[0])]
            trIou.append(float(data[i].split("|")[-1].split("/n")[0]))
        
        if ("Validation" in data[i]) and ("IoU" in data[i-2]):
            for j,d in enumerate(data[i].split("|")[1:-1]):
                try:
                    testIou[LABEL_TO_ORGAN[j]].append(float(d.split("/n")[0]))
                except:
                    testIou[LABEL_TO_ORGAN[j]]=[float(d.split("/n")[0])]
            tstIou.append(float(float(data[i].split("|")[-1].split("/n")[0])))
    return trainLoss,testLoss,trAcc,tstAcc,trIou,tstIou,trainAcc,trainIou,testAcc,testIou



# Functions for plotting the arrays extracted from .out filesa
def loss_plot(trainLoss,testLoss,save=False):
    plt.figure(figsize=(10,10))
    plt.plot(trainLoss)
    plt.plot(testLoss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend(["Train","Test"])
    plt.title("Cross Entropy Loss(Weighted)")
    if save:
        plt.savefig("Loss plot.png")
    plt.show()
def iou_plot(trainIou,testIou,save=False):
    plt.figure(figsize=(10,10))
    plt.plot(trainIou)
    plt.plot(testIou)
    plt.xlabel("Epochs")
    plt.ylabel("Iou Value")
    plt.legend(["Train","Test"])
    plt.title("Iou Plot")
    if save:
        plt.savefig("iou plot.png")
    plt.show()
def accuracy_plot(trainAcc,testAcc,save=False):
    plt.figure(figsize=(10,10))
    plt.plot(trainAcc)
    plt.plot(testAcc)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Train","Test"])
    plt.title("Accuracy Plot ")
    if save:
        plt.savefig("Accuracy plot.png")
    plt.show()

def organ_plot(train,type="train",metrics="acc",save=False):
    plt.figure(figsize=(10,10))
    plt.plot(train['background'])
    plt.plot(train['liver'])
    plt.plot(train['lungs'])
    plt.plot(train['bladder'])
    plt.plot(train['left_kidney'])
    plt.plot(train['right_kidney'])
    plt.legend(list(train.keys()))
    plt.xlabel("Epochs")
    if metrics=="acc":
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy of organs during {type}ing")
    else:
        plt.ylabel("IoU")
        plt.title("IoU of organs during training")
    
    if save:
        plt.savefig(f"organ_plot_{type}_{metrics}.png")
    
    plt.show()
