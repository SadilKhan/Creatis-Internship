import numpy as np
import pandas as pd
import nibabel as nib
import os
import matplotlib.pyplot as plt

SEG_MASK=np.array(os.listdir("/home/khan/Internship/Dataset/visceral/segmentations"))

radLexIDDict={1247:"trachea",1302:"right lung",1326:"left lung",170:" pancreas",
187:"gallbladder",237:"urinary bladder",2473: "sternum",29193:"firdt lumber vertebra",
29662:"right kidney",29663:"left kidney",30324:"right adrenal gland",
30325:"left adrenal gland",32248:"right psoas major",32249:"left psoas major",
40357:"right rectus abdominis",40358: "left rectus abdominis",480:"aorta",
58: "liver",7578:"thyroid gland",86:"spleen"}

ORGAN_CHOICE={"liver":1,"pancreas":2,"spleen":3,"right lung":4,"left lung":5,"left kidney":6,"right kidney":7}

def find_segmentation_mask(patNum):
    """ Finds all the segmentation masks according to Patient Id"""

    global allSegMask
    # Split the segmentation mask strings
    allSegMaskSpl=[i.split("_") for i in allSegMask]
    
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


def transformToRas(imageName):
    """ Function to transform Image orientation to RAS """

    image=nib.load(imageName)
    orientation=nib.aff2axcodes(image.affine)

    if orientation!=('R', 'A', 'S'):
        fimage=nib.as_closest_canonical(image)
        return fimage
    else:
        print("IMAGE ALREADY IN RAS+ FORMAT")
        return image



def findCorners(segImageSlice):
    """ Finds the bounding box of a 2d organ image"""
    x,y=(segImageSlice>0).nonzero()

    [x1,y1]=[x[0],np.min(y)]
    [x2,y2]=[x[-1],np.max(y)]

    return [x1,y1],[x2,y2]

def findCube(segImage):
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
        A,B=findCorners(segImage[:,:,i])

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


def samplePoints(point1,point2,num=2000):
    """ Uniform Point Sampling """

    points=np.linspace(point1,point2,num)
    points=np.round(points)
    points=points.astype(int)

    return points
