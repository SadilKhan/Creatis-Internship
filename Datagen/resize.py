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

    def __init__(self,datatype,imagePath,resizeRate):
        self.datatype = datatype
        self.imagePath = imagePath
        self.resizeRate = resizeRate

        self.resize()
    
    def resize(self):
        image=nib.load(self.imagePath)
        imageData=image.get_fdata()

        # Resize the image
        rescaledImageData=zoom(imageData, self.resizeRate)
        nib.save(nib.Nifti1Image(rescaledImageData,image.affine,image.header),self.imagePath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datatype",help="ctorg or visceral",type=str,default="visceral")
    parser.add_argument("--inputDir",help="Image Directory",type=str,required=True)
    parser.add_argument("--resizeRate",help="Resize rate",type=float,nargs=3,default=[0.25,0.25,0.25])

    args=parser.parse_args()
    
    if args.datatype=="visceral":
        allImagePath=glob.glob(os.path.join(args.inputDir,"*"))
    else:
        allImageName=ctorg_find_volume_mask(args.inputDir)
        allImagePath=[args.inputDir+"/"+i for i in allImageName]
    for path in tqdm(allImagePath):
        ResizeImage(args.datatype,path,args.resizeRate)
            



if __name__ == "__main__":
    main()

        
        
