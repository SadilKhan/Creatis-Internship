import argparse
import numpy as np
import nibabel as nib
from utils import *

def gradient(imageName,save=False,outputDir=None):
    """ Calculation of gradient values """

    image=transform_to_ras(imageName)
    imageData=image.get_fdata()

    gradx,grady,gradz=np.gradient(imageData)

    magnitude=np.sqrt(gradx**2+grady**2+gradz**2+gradz**2)

    
    if save:
        outputMagName=outputDir+"_mag"+imageName.split('.',maxsplit=1)[1]
        outputGradxName=outputDir+"_gradx"+imageName.split('.',maxsplit=1)[1]
        outputGradyName=outputDir+"_grady"+imageName.split('.',maxsplit=1)[1]
        outputGradzName=outputDir+"_gradz"+imageName.split('.',maxsplit=1)[1]

        nib.save(nib.Nifti1Image(magnitude,np.eye(4)),outputMagName)
        nib.save(nib.Nifti1Image(gradx,np.eye(4)),outputGradxName)
        nib.save(nib.Nifti1Image(grady,np.eye(4)),outputGradyName)
        nib.save(nib.Nifti1Image(gradz,np.eye(4)),outputGradzName)
    else:
        return gradx,grady,gradz,magnitude



def main():
    parser=argparse.ArgumentParser()

    parser.add_argument('--imageDir',type=str,help="Image Directory",required=True)
    parser.add_argument('--save',type=bool,default=False)
    parser.add_argument("--outputDir",type=str,help="Output Directory",default=args.imageDir)

    args=parser.parse_args()
    
    gradient(args.imageDir,args.save,args.imageDir)


if __name__=="__main__":
    main()

    
