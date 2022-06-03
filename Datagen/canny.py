import os
import itk
from tqdm import tqdm
import argparse

from utils import *


def canny(inputImage,outputImage,variance,lowerT,upperT):

    InputPixelType = itk.F
    OutputPixelType = itk.UC
    Dimension = 3

    InputImageType = itk.Image[InputPixelType, Dimension]
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    reader = itk.ImageFileReader[InputImageType].New()
    reader.SetFileName(inputImage)

    cannyFilter = itk.CannyEdgeDetectionImageFilter[
        InputImageType,
        InputImageType].New()
    cannyFilter.SetInput(reader.GetOutput())
    cannyFilter.SetVariance(variance)
    cannyFilter.SetLowerThreshold(lowerT)
    cannyFilter.SetUpperThreshold(upperT)

    rescaler = itk.RescaleIntensityImageFilter[
        InputImageType,
        OutputImageType].New()
    rescaler.SetInput(cannyFilter.GetOutput())
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)

    writer = itk.ImageFileWriter[OutputImageType].New()
    writer.SetFileName(outputImage)
    writer.SetInput(rescaler.GetOutput())

    writer.Update()


def main():
    parser = argparse.ArgumentParser(description="Positional Arguments for Canny Edge Detection")
    parser.add_argument('--datatype',type=str,help="visceral or ctorg",required=True)
    parser.add_argument('--inputDir',type=str,help="Image Directory",required=True)
    parser.add_argument('--outputDir',type=str,help="Output Directory",required=True)
    parser.add_argument('--variance',type=list,nargs=3,default=[1,1,1])
    parser.add_argument('--lowerT',type=float,default=20,help="Lower threshold for Edge Detection")
    parser.add_argument('--upperT',type=float,default=50,help="Upper threshold for Edge Detection")

    args=parser.parse_args()
    if args.datatype=="ctorg":
        inputImages=ctorg_find_volume_mask(args.inputDir)
    else:
        inputImages=os.listdir(args.inputDir)
    for img in tqdm(inputImages):
        if args.datatype=="ctorg":
            outImageName=img.split(".")[0]+f"_edge_{args.lowerT}_{args.upperT}"+".nii.gz"
        else:
            outImageName=img.split("/")[-1].split(".")[0]+f"_edge_{args.lowerT}_{args.upperT}"+".nii.gz"
        outImagePath=args.outputDir+"/"+outImageName
        print(img)
         # Edge Detection
        canny(args.inputDir+"/"+img,outImagePath,args.variance,args.lowerT,args.upperT)


   


if __name__ == "__main__":
    main()
