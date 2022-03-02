import numpy as np
import pandas as pd
import nibabel as nib
import os
import matplotlib.pyplot as plt

SEG_MASK=np.array(os.listdir("/home/khan/Internship/Dataset/visceral/segmentations"))

def find_segmentation_mask(segDir,patNum):
    """ Finds all the segmentation masks according to Patient Id"""
    
    # Get all the segmentation names
    segMaskNames=os.listdir(segDir)

    # Split the segmentation mask strings
    allSegMaskSpl=[i.split("_") for i in segMaskNames]
    
    # The first item is the patient number
    allIndex=[i for i in range(len(allSegMaskSpl)) if allSegMaskSpl[i][0]==str(patNum)]
    return allSegMask[allIndex]

def seg_to_class(segMask):
    """ Seg ID to Class Name"""
    Id=segMask.split("_")[-2]

    try:
        return radLexIDDict[Id]
    except:
        return "background"

def visualize(image,segmask):
    """ Plot Segmentation Mask """
    plt.figure(figsize=(10,10))
    plt.imshow(image,cmap="gray")
    plt.imshow(segmask,cmap="jet",alpha=0.25)
    plt.show()

def find_pos_organ(segmaskDir):
    """ Finds 3d Coordinates of the segmentation mask"""

    segMask=nib.load(segmaskDir)
    segMaskArr=segMask.get_fdata()

    dimension=segMaskArr.shape
    posList=[]

    for i in range(dimension[-1]):
        segLayer=segMaskArr[:,:,i]
        posx,posy=(segLayer>0).nonzero()
        posList+=list(map(lambda x,y:np.array([x,y,i]),posx,posy))
    return posList

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


def sample_points(point1,point2,num=2000):
    """ Uniform Point Sampling """

    points=np.linspace(point1,point2,num)
    points=np.round(points)
    points=points.astype(int)

    return points

def filter_points(points,segImage):
    """ Filter those Points which are in segmentation mask """

    filteredPoints=[]

    for x,y,z in points:
        if segImage[x,y,z]>0:
            filteredPoints.append([x,y,z ])
    
    return np.array(filteredPoints)
