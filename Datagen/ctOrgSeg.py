import numpy as np
import nibabel as nib
from torch import threshold
from utils import transform_to_ras
import argparse
import os
from tqdm import tqdm

def corSegLabel(segFile,threshold):
    """ Function for creating labels from kidney to right and left kidney """
    seg=nib.load(segFile)
    segData=seg.get_fdata()
    segData=np.ceil(segData)
    
    # Find positions for kidney
    xk,yk,zk=(segData==4).nonzero()
    
    # Find left right kidney positions
    # threshold for the distance between two consecutive positions
    for i in range(1,len(xk)):
        if np.abs(xk[i]-xk[i-1])>threshold:
            break    
    # The positions for the right kidney
    xkr,ykr,zkr=xk[i:],yk[i:],zk[i:]

    for i in range(len(xkr)):
        segData[xkr[i],ykr[i],zkr[i]]=7
    """segName=segFile.split("/")[-1].split(".")
    segName[0]+="corrected"
    segName="/".join(segFile.split("/")[:-1])+"/"+".".join(segName)
    """
    nib.save(nib.Nifti1Image(segData,seg.affine),segFile)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segDir",help="The directory containing segmentation masks",required=True)
    parser.add_argument("--threshold",help="threshold for the distance between two consecutive positions",type=float,default=10)

    args=parser.parse_args()

    segDir=args.segDir
    segMasks=os.listdir(segDir)

    for msk in tqdm(segMasks):
        if "labels" in msk:
            corSegLabel(segDir+"/"+msk,args.threshold)


if __name__ == "__main__":
    main()
