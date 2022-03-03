import numpy as np
import nibabel as nib
from Datagen.utils import transform_to_ras
from globalVar import *
from utils import *
import pandas as pd
import argparse
from tqdm import tqdm
from calGradient import gradient


class PointCloudGen():
    """ Class for sampling point clouds from voxel data """

    def __init__(self,imageDir,segDir,outputDir):
        """

        @input imageDir: string like. The path for specific image
        @input segDir: string like. The directory for Segmentations

        """
        self.imageDir=imageDir
        self.segDir=segDir
        self.outputDir=outputDir

        self.gradx,self.grady,self.gradz,self.mag=gradient(self.imageDir)

        self.patNum=self.imageDir.split("/")[-1].split("_")[0]


        # Initialize and load necessary variables
        self.initialize()
        # Generate Point cloud for every organ
        self.generate_point_cloud()

        # Save the point clouds in csv file
        self.save()


    def initialize(self):
        self.image=transform_to_ras(self.imageDir)
        self.imageData=self.image.get_fdata()

        self.organPC=pd.DataFrame({"x":[],"y":[],"z":[],"value":[],"magnitude":[],"label":[]})
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
            segMask=transform_to_ras(self.segDir+"/"+name)
            segMaskData=segMask.get_fdata()

            cube=find_cube(segMaskData)

            points=sample_points(cube[0],cube[1])
            filteredPoints=embed_points(points,self.imageData,segMaskData,self.mag,0.15,organ)

            self.organPC=pd.concat([self.organPC,pd.DataFrame({"x":filteredPoints[:,0],
            "y":filteredPoints[:,1],"z":filteredPoints[:,2],"value":filteredPoints[:,3],"magnitude":filteredPoints[:,4],
            "label":filteredPoints[:,5]})])


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
        print(imName)
        pc=PointCloudGen(IMAGE_PATH+"/"+imName,args.segDir,args.outputDir)



if __name__== "__main__":
    main()



        

        

                    



