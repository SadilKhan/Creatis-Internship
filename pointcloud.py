import numpy as np
import nibabel as nib
from globalVar import *
from utils import *
import pandas as pd



class PointCloudGen():
    """ Class for sampling point clouds from voxel data """

    def __init__(self,imageDir,segDir):
        """

        @input imageDir: string like. The Directory for image
        @input segDir: string like. The directory for Segmentations

        """
        self.imageDir=imageDir
        self.segDir=segDir

        # Initialize and load necessary variables
        self.initialize()
        # Generate Point cloud for every organ
        self.generate_point_cloud()

        # Save the point clouds in csv file
        self.save()


    def initialize(self):
        self.image=nib.load(self.imageDir)
        self.imageData=self.image.get_fdata()

        self.organPC=pd.DataFrame({"x":[],"y":[],"z":[],"label":[]})
        self.segMask=dict()

    def generate_point_cloud(self):
        allSegName=os.listdir(self.segDir)
        formatSegName=allSegName[0].split("_")

        # Get segmentation data for every organ and save it in dictionary
        for organ in ORGAN_CHOICE.keys():
            codeOrgan=str(ORGAN_CHOICE[organ])  
            formatSegName[-2]=codeOrgan
            segMaskName="_".join(formatSegName)
            segMask=nib.load(segMaskName)
            segMaskData=segMask.get_fdata()

            cube=find_cube(segMaskData)

    def save(self):
        pass


        

        

                    




