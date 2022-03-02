import numpy as np
import nibabel as nib
from globalVar import *
from utils import *
import pandas as pd
import argparse



class PointCloudGen():
    """ Class for sampling point clouds from voxel data """

    def __init__(self,imageDir,segDir):
        """

        @input imageDir: string like. The Directory for image
        @input segDir: string like. The directory for Segmentations

        """
        self.imageDir=imageDir
        self.segDir=segDir

        self.patNum=self.imageDir.split("/")[-1].split("_")[0]


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
        self.allSegName=find_segmentation_mask(self.segDir,self.patNum)
        formatSegName=self.allSegName[0].split("_")

        # Get point clouds for every organs and store it in Dataframe
        for organ in ORGAN_CHOICE.keys():
            codeOrgan=str(ORGAN_CHOICE[organ])  
            formatSegName[-2]=codeOrgan
            segMaskName="_".join(formatSegName)
            segMask=nib.load(self.segDir+"/"+segMaskName)
            segMaskData=segMask.get_fdata()

            cube=find_cube(segMaskData)

            points=sample_points(cube[0],cube[1])
            filtedPoints=filter_points(points,segMaskData,0.2,organ)
            self.organPC=pd.concat([self.organPC,pd.DataFrame({"x":filtedPoints[:,0],
            "y":filtedPoints[:,1],"z":filtedPoints[:,2],"label":filtedPoints[:,3]})])


    def save(self):

        directory=self.segDir+"/"+self.patNum+"_point_cloud.csv"
        self.organPC.to_csv(directory,index=False)


def main():

    parser=argparse.ArgumentParser(description="Arguments for Point Cloud Generation")

    parser.add_argument("--imageDir",help="Image Path", required=True)
    parser.add_argument("--segDir",help="Path for Segmentation Mask",required=True)

    args=parser.parse_args()

    pc=PointCloudGen(args.imageDir,args.segDir)



if __name__== "__main__":
    main()



        

        

                    



