import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

SEG_MASK=np.array(os.listdir("/home/khan/Internship/Dataset/visceral/segmentations"))

radLexIDDict={1247:"trachea",1302:"right lung",1326:"left lung",170:" pancreas",
187:"gallbladder",237:"urinary bladder",2473: "sternum",29193:"firdt lumber vertebra",
29662:"right kidney",29663:"left kidney",30324:"right adrenal gland",
30325:"left adrenal gland",32248:"right psoas major",32249:"left psoas major",
40357:"right rectus abdominis",40358: "left rectus abdominis",480:"aorta",
58: "liver",7578:"thyroid gland",86:"spleen"}


ORGAN_CHOICE={"liver","pancreas","spleen","right lung",
"left lung","left kidney","right kidney"}

def find_segmentation_mask(patNum):

    global allSegMask
    # Split the segmentation mask strings
    allSegMaskSpl=[i.split("_") for i in allSegMask]
    
    # The first item is the patient number
    allIndex=[i for i in range(len(allSegMaskSpl)) if allSegMaskSpl[i][0]==str(patNum)]
    return allSegMask[allIndex]

def seg_to_class(segMask):
    Id=segMask.split("_")[-2]

    try:
        return radLexIDDict[Id]
    except:
        return "background"

def visualize(image,segmask):
    plt.figure(figsize=(10,10))
    plt.imshow(image,cmap="gray")
    plt.imshow(segmask,cmap="jet",alpha=0.25)
    plt.show()

def find_pos_organ(segmask):
    """ Finds 3d Coordinates of the segmentation mask"""

    dimension=segmask.shape
    posList=[]

    for i in range(dimension[-1]):
        segLayer=segmask[:,:,i]
        posx,posy=(segLayer>0).nonzero()
        posList+=list(map(lambda x,y:np.array([x,y,i]),posx,posy))
    return posList
