import argparse
import numpy as np
import nibabel as nib
from utils import transform_to_ras

def gradient(imageDir,save=False):

    image=transform_to_ras(imageDir)
    imageData=image.get_fdata(image)


    gradx,grady,gradz=np.gradient(imageData)

    magnitude=np.sqrt(gradx**2+grady**2+gradz**2+gradz**2)

    outputMagName=imageDir.split('.',maxsplit=1)[0]+"_mag"+imageDir.split('.',maxsplit=1)[1]
    outputGradxName=imageDir.split('.',maxsplit=1)[0]+"_gradx"+imageDir.split('.',maxsplit=1)[1]
    outputGradyName=imageDir.split('.',maxsplit=1)[0]+"_grady"+imageDir.split('.',maxsplit=1)[1]
    outputGradzName=imageDir.split('.',maxsplit=1)[0]+"_gradz"+imageDir.split('.',maxsplit=1)[1]

    if save:
        nib.save(nib.Nifti1Image(magnitude,np.eye(4)),outputMagName)
        nib.save(nib.Nifti1Image(gradx,np.eye(4)),outputGradxName)
        nib.save(nib.Nifti1Image(grady,np.eye(4)),outputGradyName)
        nib.save(nib.Nifti1Image(gradz,np.eye(4)),outputGradzName)
    else:
        return gradx,grady,gradz,magnitude



def main():
    parser=argparse.ArgumentParser()

    parser.add_argument('--imageDir',type=str,help="Image Directory",required=True)

    args=parser.parse_args()

    gradient(args.imageDir)


if __name__=="__main__":
    main()

    
