import numpy as np
import nibabel as nib
from globalVar import *
from utils import *
import pandas as pd
import argparse
from tqdm import tqdm
from calGradient import gradient
import time


class PointCloudGen():
    """ Class for sampling point clouds from voxel data """

    def __init__(self,edgePath,segDir,outputDir):
        """
        @input edgePath: string like. The path for specific edge detected image
        @input segDir: string like. The directory for Segmentations
        @input outputDir: string like. The output Directory for csv file
        """

        self.edgePath=edgePath
        self.segDir=segDir
        self.outputDir=outputDir

        self.patNum=self.edgePath.split("/")[-1].split("_")[0]


        start=time.time()
        #Initialize Variable
        self.initialize()

        # Generate point clouds
        self.generate_point_cloud()

        # Save the point clouds in csv file
        self.save()

        end=time.time()
        print(f"TOTAL TIME TAKEN: {(end-start)//60} minutes {(end-start)%60} secs")
    
    def initialize(self):
        self.image=transform_to_ras(self.edgePath)
        self.imageData=self.image.get_fdata()
        self.affineMat=self.image.affine
        self.rotMat=self.affineMat[:3,:3]
        self.transMat=self.affineMat[:3,3]

        
        self.segMask=dict()
    def generate_point_cloud(self):
        self.allSegName=find_segmentation_mask(self.segDir,self.patNum)
        orgToSeg=dict()

        # Get positions for edges
        X,Y,Z=(self.imageData>0).nonzero()

        self.organPC=pd.DataFrame({"x":X,"y":Y,"z":Z,"label":[None]*len(X)})


        # Get point cloud labels for every organs and store it in Dataframe
        for organ in tqdm(ORGAN_CHOICE.keys()):
            codeOrgan=f"_{ORGAN_CHOICE[organ]}_"
            # Find segmentation mask for organ with patientID self.patNum
            for name in self.allSegName:
                if codeOrgan in name:
                    break
            segMask=transform_to_ras(self.segDir+"/"+name)
            segMaskData=segMask.get_fdata()
            
            # Dilate the segmentation mask onces
            segMaskData=dilation(segMaskData)

            # Segmentation Mask positions
            X_seg,Y_seg,Z_seg=(segMaskData>0).nonzero()

            # Create a dataframe for segmentation
            organDF=pd.DataFrame({"x":X_seg,"y":Y_seg,"z":Z_seg,"label":[organ]*len(X_seg)})
            
            # Get the label for the position that is in segmentation mask and an edge
            self.organPC=  pd.merge(self.organPC,organDF,how="left",on=["x","y","z"])

            self.organPC['label']=self.organPC["label_x"].combine_first(self.organPC["label_y"])
            del self.organPC["label_x"]
            del self.organPC["label_y"]
        self.organPC['label']=self.organPC['label'].fillna("Background")

    def save(self):

        if self.outputDir=="default":
            directory=self.segDir
        else:
            directory=self.outputDir
        
        directory+="/"+self.patNum+"_point_cloud.csv"
        self.organPC.to_csv(directory,index=False)



def main():

    parser=argparse.ArgumentParser(description="Arguments for Point Cloud Generation")

    parser.add_argument("--edgeDir",help="Edge Directory", required=True)
    parser.add_argument("--segDir",help="Segmentation Directory",required=True)
    parser.add_argument("--outputDir",help="Output Directory",default="default")

    args=parser.parse_args()

    IMAGE_PATH=args.edgeDir
    allImages=os.listdir(IMAGE_PATH)

    for imName in tqdm(allImages):
        print(imName)
        pc=PointCloudGen(IMAGE_PATH+"/"+imName,args.segDir,args.outputDir)



if __name__== "__main__":
    main()
