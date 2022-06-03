import nibabel as nib
import os
import glob
from utils import *
def flipAndSaveToRAS(fileName,outputDir):
    """
    Function used to flip the orientation of the original data to RAS+ orientation
    Save the data flipped under the same name into a sub-directory called "RASData"
    @input filename : path of the file to flip to RAS
    """
    
    #Recover the image object
    imageObj = nib.load(fileName)
    
    #Get the current orientation
    CurrentOrientation = nib.aff2axcodes(imageObj.affine)
    print("The current orientation is : ", CurrentOrientation)
    
    #Check if the current orientation is already RAS+
    if CurrentOrientation == ('R', 'A', 'S') :
        
        print("Image already recorded into the RAS+ orientation, nothing to do")
        nib.save(imageObj,outputDir+"/"+fileName.split("/")[-1])
        
    else :
        #Flip the image to RAS
        flippedImage = nib.as_closest_canonical(imageObj)
                
        ##Check the new orientation
        NewOrientation = nib.aff2axcodes(flippedImage.affine)
        
        #Set Qcode to 1 that the Qform matrix can be used into the further processing
        flippedImage.header['qform_code'] = 1
        
        #Save the flipped image
        nib.save(flippedImage, outputDir+"/"+fileName.split("/")[-1])
        
        print("The new orientation is now : ", NewOrientation)
        
        #########
        ## Test #
        #########
        
        ###Check if the we saved the RAS+ data
        imTest = nib.load(outputDir+"/"+fileName.split("/")[-1])
        print(nib.aff2axcodes(imTest.affine), imTest.get_qform(), imTest.header['qform_code'])
                

    
def main():
    """
    Function to run as main routine
    """
    import argparse
    
    parser = argparse.ArgumentParser(description = 'Flip the orientation of the file (-f) or directory (-d) to RAS+ and save it into a sub-directory called RASData')
    parser.add_argument('--dataset_type',help="visceral/ctorg",type=str,default="visceral")
    parser.add_argument('--imageDir',  help = 'Image Directory')
    parser.add_argument('--outputDir', help = 'Output Directory')
    args = parser.parse_args()
    
    if args.dataset_type=="visceral":
        imagePath=glob.glob(os.path.join(args.imageDir,'*'))
    else:
        imageName=ctorg_find_volume_mask(args.imageDir)
        imagePath=[args.imageDir+"/"+i for i in imageName]

    for path in imagePath:
        flipAndSaveToRAS(path,args.outputDir)
    
if __name__ == '__main__':
    main()
