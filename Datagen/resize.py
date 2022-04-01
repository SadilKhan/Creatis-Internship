import numpy as np
import glob
import nibabel as nib
import os
import argparse
from scipy.ndimage import zoom
from utils import *
from tqdm import tqdm


class ResizeImage:
    """ Custom Class for Image Resize """

    def __init__(self,datatype,imagePath,segPath,resizeRate):
        self.datatype = datatype
        self.imagePath = imagePath
        self.segPath = segPath
        self.resizeRate = resizeRate

        if self.datatype == "visceral":
            self.Num = self.imagePath.split("/")[-1].split("_")[0]
            allSegName= find_segmentation_mask(self.segPath,self.Num)
            self.allSegPath=[self.segPath+"/"+i for i in allSegName]
        else:
            self.Num = self.imagePath.split("/")[-1].split("_")[0].split("-")[-1]
            allSegName= ctorg_find_seg_mask(self.segPath,self.Num)
            self.allSegPath=[self.segPath+"/"+i for i in allSegName]

        self.resize()
    
    def resize(self):
        image=nib.load(self.imagePath)
        imageData=image.get_fdata()

        # Resize the image
        rescaledImageData=zoom(imageData, self.resizeRate)
        nib.save(nib.Nifti1Image(rescaledImageData,image.affine),self.imagePath)

        # Resize the Segmentation
        for path in tqdm(self.allSegPath):
            #print(path)
            seg=nib.load(path)
            segData=seg.get_fdata()
            rescaledSegData=zoom(segData, self.resizeRate)
            nib.save(nib.Nifti1Image(rescaledSegData,seg.affine),path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datatype",help="ctorg or visceral",type=str,default="visceral")
    parser.add_argument("--inputDir",help="Image Directory",type=str,required=True)
    parser.add_argument("--segDir",help="Segmentation Directory",type=str,required=True)
    parser.add_argument("--resizeRate",help="Resize rate",type=float,nargs=3,default=[0.25,0.25,0.25])

    args=parser.parse_args()

    allImagePath=glob.glob(os.path.join(args.inputDir,"*"))

    for path in tqdm(allImagePath):
        ResizeImage(args.datatype,path,args.segDir,args.resizeRate)
            



if __name__ == "__main__":
    main()

        
        
