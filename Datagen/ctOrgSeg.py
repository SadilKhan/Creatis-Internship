import numpy as np
import nibabel as nib
from torch import threshold
from utils import ctorg_find_seg_mask, transform_to_ras
import argparse
import os
from utils import *
from tqdm import tqdm

def corSegLabel(segFile):
    """ Function for creating labels from kidney to right and left kidney """
    seg=nib.load(segFile)
    segData=seg.get_fdata()
    segData=np.ceil(segData)
    
    # Find positions for kidney
    xk,yk,zk=(segData==4).nonzero()

    # Create bounding cube
    A0,A1=find_cube(segData,label=4)

    hyperplane=(A0+A1)/2

    # Find left right kidney positions
    x_index=(xk<hyperplane[0]).nonzero()[0].max()
    
    # The positions for the right kidney
    xkr,ykr,zkr=xk[x_index+1:],yk[x_index+1:],zk[x_index+1:]

    for i in range(len(xkr)):
        segData[xkr[i],ykr[i],zkr[i]]=7
    
    nib.save(nib.Nifti1Image(segData,seg.affine),segFile)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segDir",help="The directory containing segmentation masks",required=True)
    args=parser.parse_args()

    segDir=args.segDir
    segMasks=ctorg_find_seg_mask(segDir)

    for msk in tqdm(segMasks):
        corSegLabel(segDir+"/"+msk)


if __name__ == "__main__":
    main()
