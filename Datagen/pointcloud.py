import numpy as np
import nibabel as nib
from globalVar import *
from utils import *
import pandas as pd
import argparse
from tqdm import tqdm



class PointCloudGen():
    """ Class for sampling point clouds from voxel data """

    def __init__(self,imageDir,segDir,outputDir):
        """

        @input imageDir: string like. The Directory for image
        @input segDir: string like. The directory for Segmentations
        @input outputDir: string like. The directory where csv file be stored

        """
        self.imageDir=imageDir
        self.segDir=segDir
        self.outputDir=outputDir

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
        orgToSeg=dict()

        # Get point clouds for every organs and store it in Dataframe
        for organ in ORGAN_CHOICE.keys():
            codeOrgan=f"_{ORGAN_CHOICE[organ]}_"
            # Find segmentation mask for organ with patientID self.patNum
            for name in self.allSegName:
                if codeOrgan in name:
                    break
            segMask=nib.load(self.segDir+"/"+name)
            segMaskData=segMask.get_fdata()

            cube=find_cube(segMaskData)

            points=sample_points(cube[0],cube[1])
            filtedPoints=filter_points(points,segMaskData,0.2,organ)
            self.organPC=pd.concat([self.organPC,pd.DataFrame({"x":filtedPoints[:,0],
            "y":filtedPoints[:,1],"z":filtedPoints[:,2],"label":filtedPoints[:,3]})])


    def save(self):

        if self.outputDir=="default":
            directory=self.segDir
        else:
            directory=self.outputDir
        
        directory+="/"+self.patNum+"_point_cloud.csv"
        self.organPC.to_csv(directory,index=False)


def main():

    parser=argparse.ArgumentParser(description="Arguments for Point Cloud Generation")

    parser.add_argument("--imageDir",help="Image Directory", required=True)
    parser.add_argument("--segDir",help="Segmentation Directory",required=True)
    parser.add_argument("--outputDir",help="Output Directory",default="default")

    args=parser.parse_args()

    IMAGE_PATH=args.imageDir
    allImages=os.listdir(IMAGE_PATH)

    for imName in tqdm(allImages):
        pc=PointCloudGen(IMAGE_PATH+"/"+imName,args.segDir,args.outputDir)



if __name__== "__main__":
    main()



        

        

                    



