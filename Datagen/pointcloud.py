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
    """ Class for sampling point clouds from visceral or ctOrg dataset"""

    def __init__(self, datatype, edgePath, segDir, outputDir):
        """
        @input datatype: string like. Visceral, CtOrg
        @input edgePath: string like. The path for specific edge detected image
        @input segDir: string like. The directory for Segmentations
        @input outputDir: string like. The output Directory for csv file
        """

        self.datatype = datatype
        self.edgePath = edgePath
        self.segDir = segDir
        self.outputDir = outputDir

        if self.datatype == "visceral":
            self.Num = int(self.edgePath.split("/")[-1].split("_")[0])
        else:
            self.Num = int(self.edgePath.split("/")[-1].split("_")[0].split("-")[-1])

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
        self.image = transform_to_ras(self.edgePath)
        self.imageData = self.image.get_fdata()
        self.affineMat = self.image.affine
        self.rotMat = self.affineMat[:3, :3]
        self.transMat = self.affineMat[:3, 3]

        self.segMask = dict()

    def generate_point_cloud_visceral(self):
        """ Method for generating point clouds for visceral dataset """
        self.allSegName = find_segmentation_mask(self.segDir, self.Num)
        orgToSeg = dict()

        # Get positions for edges
        X, Y, Z = (self.imageData > 0).nonzero()

        self.organPC = pd.DataFrame(
            {"x": X, "y": Y, "z": Z, "label": [None]*len(X)})

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
        # Replace the left lung and right lung into single label lungs
        self.organPC['label']=self.organPC['label'].replace({"left_lung":"lungs","right_lung":"lungs"})

        # From Voxel Space to Scanner Space
        self.organPC[['x', 'y', 'z']] = self.organPC[[
            'x', 'y', 'z']].values@self.rotMat.T+self.transMat

    def generate_point_cloud_ctorg(self):
        """ Method for generating point clouds for Ctorg dataset """
        self.segName = ctorg_find_seg_mask(self.segDir, self.Num)
        segMask = transform_to_ras(self.segDir+"/"+self.segName)
        segMaskData = segMask.get_fdata()
        #segMaskData = np.ceil(segMaskData)
        orgToSeg = dict()

        # Get positions for edges
        X, Y, Z = (self.imageData > 0).nonzero()

        self.organPC = pd.DataFrame(
            {"x": X, "y": Y, "z": Z, "label": [None]*len(X)})
        # Get point cloud labels for every organs and store it in Dataframe
        for organ in CTORG_ORGAN_CHOICE.keys():
            labelOrgan = CTORG_ORGAN_CHOICE[organ]
            segMaskDataTemp=(segMaskData==labelOrgan)*1
            segMaskDataTemp=dilation(segMaskDataTemp)

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

        # From Voxel Space to Scanner Space
        self.organPC[['x', 'y', 'z']] = self.organPC[[
            'x', 'y', 'z']].values@self.rotMat.T+self.transMat

    def save(self):

        if self.outputDir == "default":
            directory = self.segDir
        else:
            directory = self.outputDir

        directory += "/"+self.Num+"_point_cloud.csv"
        self.organPC.to_csv(directory, index=False)


def main():

    parser = argparse.ArgumentParser(
        description="Arguments for Point Cloud Generation")

    parser.add_argument(
        "--datatype", help="visceral or ctorg", type=str, required=True)
    parser.add_argument("--edgeDir", help="Edge Directory", required=True)
    parser.add_argument(
        "--segDir", help="Segmentation Directory", required=True)
    parser.add_argument(
        "--outputDir", help="Output Directory", default="default")

    args = parser.parse_args()

    IMAGE_PATH = args.edgeDir
    allImages = os.listdir(IMAGE_PATH)
    for imName in tqdm(allImages):
        # print(imName)
        pc = PointCloudGen(args.datatype, IMAGE_PATH+"/" +
                           imName, args.segDir, args.outputDir)


if __name__ == "__main__":
    main()
