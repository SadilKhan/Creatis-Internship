import numpy as np
import nibabel as nib
from globalVar import *
from utils import *
import pandas as pd
import argparse
from tqdm import tqdm
from calGradient import gradient
import time
from surfacePointExtractor import *


class PointCloudGen():
    """ Class for sampling point clouds from visceral or ctOrg dataset"""

    def __init__(self, datatype, edgePath, imageDir, segDir,multiplier,nbr_fs, outputDir):

        self.datatype = datatype
        self.edgePath = edgePath
        self.imageDir = imageDir
        self.segDir = segDir
        self.multiplier = multiplier
        self.nbr_fs = nbr_fs
        self.outputDir = outputDir

        if self.datatype == "visceral":
            self.Num = self.edgePath.split("/")[-1].split("_")[0]
        else:
            self.Num = self.edgePath.split("/")[-1].split("_")[0].split("-")[-1]

        # Get the image path
        self.imagePath="_".join(self.edgePath.split("/")[-1].split("_")[:-3])+".nii.gz"

        start = time.time()
        # Initialize Variable
        self.initialize()
        # Generate point clouds
        if self.datatype == "visceral":
            self.generate_point_cloud_visceral()
        else:
            self.generate_point_cloud_ctorg()

        # Save the point clouds in csv file
        self.save()

        end = time.time()
        print(
            f"TOTAL TIME TAKEN: {(end-start)//60} minutes {(end-start)%60} secs")

    def initialize(self):
        """ Initializes required variables"""
        self.edge = nib.load(self.edgePath)
        self.edgeData = self.edge.get_fdata()
        self.affineMat = self.edge.affine
        self.rotMat = self.affineMat[:3, :3]
        self.transMat = self.affineMat[:3, 3]
        self.image=nib.load(self.imageDir+"/"+self.imagePath)
        self.imageData=self.image.get_fdata()
        _,_,_,self.magnitude=gradient(self.imageDir+"/"+self.imagePath)

        self.segMask = dict()

    def generate_point_cloud_visceral(self):
        """ Method for generating point clouds for visceral dataset """
        self.allSegName = find_segmentation_mask(self.segDir, self.Num)
        orgToSeg = dict()

        # Get positions for edges
        X, Y, Z = (self.edgeData > 0).nonzero()
        intensity=self.imageData[X, Y, Z]
        gradient=self.magnitude[X, Y, Z]
        X,Y,Z=X*self.multiplier,Y*self.multiplier,Z*self.multiplier

        self.organPC = pd.DataFrame(
            {"x": X, "y": Y, "z": Z, "label": [None]*len(X),"intensity":intensity,"gradient":gradient})
        #bdPoints=pd.DataFrame({'x':[],'y':[],'z':[]})

        # Get point cloud labels for every organs and store it in Dataframe
        for organ in tqdm(ORGAN_CHOICE.keys()):
            codeOrgan = f"_{ORGAN_CHOICE[organ]}_"
            # Find segmentation mask for organ with patientID self.patNum
            for name in self.allSegName:
                if codeOrgan in name:
                    break
            segMask = transform_to_ras(self.segDir+"/"+name)
            segMaskData = segMask.get_fdata()

            # Dilate the segmentation mask onces
            segMaskData = dilation(segMaskData)
            #bdPoints=pd.concat([bdPoints,extractSurfacePoints(segMaskData)]) # Add all the background points

            # Segmentation Mask positions
            X_seg, Y_seg, Z_seg = (segMaskData > 0).nonzero()

            # Create a dataframe for segmentation
            organDF = pd.DataFrame(
                {"x": X_seg, "y": Y_seg, "z": Z_seg, "label": [organ]*len(X_seg)})

            # Get the label for the position that is in segmentation mask and an edge
            self.organPC = pd.merge(
                self.organPC, organDF, how="left", on=["x", "y", "z"])

            self.organPC['label'] = self.organPC["label_x"].combine_first(
                self.organPC["label_y"])
            del self.organPC["label_x"]
            del self.organPC["label_y"]
        self.organPC['label'] = self.organPC['label'].fillna("background")
        # Calculate Signed Distance Function
        #self.organPC=sdfCalculator(bdPoints,self.organPC)
        # Replace the left lung and right lung into single label lungs
        self.organPC['label']=self.organPC['label'].replace({"left_lung":"lungs","right_lung":"lungs"})
        # Transform the points from original to resized point space
        self.organPC[['x','y','z']]=self.organPC[['x','y','z']]/self.multiplier
        # The neighboring intensity points
        self.organPC['neighbor_intensity']=self.organPC[['x','y','z']].apply(lambda pt:find_neighbor_intensity(*pt,self.nbr_fs,self.imageData),axis=1)
        # Save the original point
        self.organPC[['xo', 'yo', 'zo']] = self.organPC[['x', 'y', 'z']]
        # From Voxel Space to Object Space
        self.organPC[['x', 'y', 'z']] = self.organPC[['x', 'y', 'z']].values@self.rotMat.T+self.transMat

    def generate_point_cloud_ctorg(self):
        """ Method for generating point clouds for Ctorg dataset """
        self.segName = ctorg_find_seg_mask(self.segDir, int(self.Num))
        segMask = transform_to_ras(self.segDir+"/"+self.segName)
        print(nib.aff2axcodes(segMask.affine))
        segMaskData = segMask.get_fdata()
        #segMaskData = np.ceil(segMaskData)
        orgToSeg = dict()

        # Get positions for edges
        X, Y, Z = (self.edgeData > 0).nonzero()
        intensity=self.imageData[X, Y, Z]
        gradient=self.magnitude[X, Y, Z]
        X,Y,Z=X*self.multiplier,Y*self.multiplier,Z*self.multiplier

        self.organPC = pd.DataFrame(
            {"x": X, "y": Y, "z": Z, "label": [None]*len(X),"intensity":intensity,"gradient":gradient})
        bdPoints=pd.DataFrame({'x':[],'y':[],'z':[]})
        # Get point cloud labels for every organs and store it in Dataframe
        for organ in CTORG_ORGAN_CHOICE.keys():
            labelOrgan = CTORG_ORGAN_CHOICE[organ]
            segMaskDataTemp=(segMaskData==labelOrgan)*1
            segMaskDataTemp=dilation(segMaskDataTemp)
            bdPoints=pd.concat([bdPoints,extractSurfacePoints(segMaskDataTemp)]) # Add all the background points

            # Segmentation Mask positions
            X_seg, Y_seg, Z_seg = (segMaskDataTemp == 1).nonzero()
            # Create a dataframe for segmentation
            organDF = pd.DataFrame(
                {"x": X_seg, "y": Y_seg, "z": Z_seg, "label": [organ]*len(X_seg)})

            # Get the label for the position that is in segmentation mask and an edge
            self.organPC = pd.merge(
                self.organPC, organDF, how="left", on=["x", "y", "z"])

            self.organPC['label'] = self.organPC["label_x"].combine_first(
                self.organPC["label_y"])
            del self.organPC["label_x"]
            del self.organPC["label_y"]
        self.organPC['label'] = self.organPC['label'].fillna("background")
        # Calculate Signed Distance Function
        self.organPC=sdfCalculator(bdPoints,self.organPC)
        # Transform the points from original to resized point space
        self.organPC[['x','y','z']]=self.organPC[['x','y','z']]/self.multiplier
        # The neighboring intensity points
        self.organPC['neighbor_intensity']=self.organPC[['x','y','z']].apply(lambda pt:find_neighbor_intensity(*pt,self.nbr_fs,self.imageData),axis=1)
        # Save the original point
        #self.organPC[['xo', 'yo', 'zo']] = self.organPC[['x', 'y', 'z']]
        # From Voxel Space to Object Space
        self.organPC[['x', 'y', 'z']] = self.organPC[['x', 'y', 'z']].values@self.rotMat.T+self.transMat

    def save(self):

        if self.outputDir == "default":
            directory = self.segDir
        else:
            directory = self.outputDir
        metadata=self.edgePath.split("/")[-1][:-7]		
        directory += "/"+metadata+"_point_cloud.pkl"
        self.organPC.to_pickle(directory)


def main():

    parser = argparse.ArgumentParser(
        description="Arguments for Point Cloud Generation")

    parser.add_argument(
        "--datatype", help="visceral or ctorg", type=str, required=True)
    parser.add_argument("--edgeDir", help="Edge Directory", required=True,type=str)
    parser.add_argument("--imageDir",help="Image Directory", required=True,type=str)
    parser.add_argument(
        "--segDir", help="Segmentation Directory", required=True,type=str)
    parser.add_argument("--multiplier",help="Multiplier",type=int,default=4)
    parser.add_argument("--nbr_fs",help="Feature size for number of neighbors, i.e 3x3, 5x5,7x7",type=int,default=3)
    parser.add_argument(
        "--outputDir", help="Output Directory", default="default",type=str)

    args = parser.parse_args()

    EDGE_PATH = args.edgeDir
    allEdges = os.listdir(EDGE_PATH)
    for edName in tqdm(allEdges):
        #print(edName)
        pc = PointCloudGen(args.datatype, EDGE_PATH+"/" +
                           edName,args.imageDir, args.segDir, args.multiplier, args.nbr_fs, args.outputDir)


if __name__ == "__main__":
    main()
