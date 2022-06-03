import numpy as np
import nibabel as nib
from globalVar import *
from utils import *
import pandas as pd
import argparse
import open3d as o3d
from calGradient import gradient


def check(imageName,segDir,num=20000,gradT=20,prob=0.1):

    image=transform_to_ras(imageName)
    gx,gy,gz,mag=gradient(imageName)
    imageData=image.get_fdata()

    x,y,z=(mag>gradT).nonzero()
    #points=

    affineMat=image.affine

    h,w,d=imageData.shape
    patNum=imageName.split("/")[-1].split("_")[0]
    allSegName=find_segmentation_mask(segDir,patNum)
    orgToSeg=dict()

    orgToColor=dict()


    # Get point clouds for every organs and store it in Dataframe
    for organ in ORGAN_CHOICE.keys():
        codeOrgan=f"_{ORGAN_CHOICE[organ]}_"
        # Find segmentation mask for organ with patientID self.patNum
        for name in allSegName:
            if codeOrgan in name:
                break
        segMask=transform_to_ras(segDir+"/"+name)
        segMaskData=segMask.get_fdata()
        orgToSeg[organ]=segMaskData
        orgToColor[organ]=np.random.choice(range(256), size=3)/255.0
    
    
    orgToColor["background"]=np.random.choice(range(256), size=3)/255.0
    points=sample_points([40,140,0],[470,450,320],num)
    main_points=[]
    colors=[]


    # Filter Points
    for p in points:
        if (mag[p[0],p[1],p[2]]>gradT):
            ifBackObj=True
            for org in ORGAN_CHOICE:
                if (orgToSeg[org][p[0],p[1],p[2]]>0):
                    colors.append(orgToColor[org])
                    main_points.append(p)
                    ifBackObj=False
                    break
            """if ifBackObj:
                colors.append(orgToColor["background"])"""
    
    main_points=np.array(main_points)@affineMat[:3,:3].T
    print(len(main_points))
    # Open3d Visualization
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(main_points)
    pcd.colors=o3d.utility.Vector3dVector(np.array(colors))  

    o3d.visualization.draw_geometries([pcd])



def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--imageName",required=True)
    parser.add_argument("--segDir",required=True)
    parser.add_argument("--num",type=int,default=20000)
    parser.add_argument("--gradT",type=float,default=20)
    parser.add_argument("--prob",type=float,default=0.1)


    args=parser.parse_args()

    check(args.imageName,args.segDir,args.num,args.gradT,args.prob)



if __name__=="__main__":
    main()
        