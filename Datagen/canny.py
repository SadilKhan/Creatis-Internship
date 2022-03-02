  
import SimpleITK as sitk
import sys
import os
import argparse



def canny(imageDir,outputDir,variance,lowerT,upperT):
    image = sitk.Cast(sitk.ReadImage(imageDir), sitk.sitkFloat32)

    blurfilter=sitk.SmoothingRecursiveGaussianImageFilter()
    blurfilter.SetSigma(1.4)
    blurfilter.Execute(image)

    edges = sitk.CannyEdgeDetection(image, lowerThreshold=lowerT, upperThreshold=upperT,
                                    variance=variance)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outputDir)
    writer.Execute(edges)


def main():
    parser = argparse.ArgumentParser(description="Positional Arguments for Canny Edge Detection")
    parser.add_argument('--imageDir',type=str,help="Image Directory",required=True)
    parser.add_argument('--outputDir',type=str,help="Output Directory",required=True)
    parser.add_argument('--variance',type=list,nargs=3,default=[1,1,1])
    parser.add_argument('--lowerT',type=float,default=20,help="Lower threshold for Edge Detection")
    parser.add_argument('--upperT',type=float,default=50,help="Upper threshold for Edge Detection")

    args=parser.parse_args()

    # Edge Detection
    canny(args.imageDir,args.outputDir,args.variance,args.lowerT,args.upperT)



if __name__== "__main__":
    main()



