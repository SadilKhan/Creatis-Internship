import open3d as o3d
import pandas as pd
import numpy as np
import argparse

from globalVar import ORGAN_TO_LABEL,ORGAN_TO_RGB


def visualize_point_cloud(csvPath,label="all",downsample=0):
    """ 3d visualization of point clouds """

    pointCloud=pd.read_csv(csvPath)
    if label=="no_background":
        pointCloud=pointCloud[(pointCloud['label']!="background")]
    elif label!="all":
        pointCloud=pointCloud[pointCloud['label']==label]
    unique_label=pointCloud["label"].unique()
    print(pointCloud['label'].value_counts())
    """labelDict=dict()
    colorDict=dict()
    # Make colors for every unique label
    for l in unique_label:
        labelDict[l]=list(colorDict[l]/255.0)
    print("Color Code",colorDict)"""

    # Make color lists
    colors=np.array([ORGAN_TO_RGB[l] for l in pointCloud["label"]])

    # open3d visualization
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(pointCloud[["x","y","z"]].values)
    pcd.colors=o3d.utility.Vector3dVector(colors)

    # Voxel Downsampling
    if downsample>0:
        print("BEFORE DOWNSAMPLING",len(pcd.points))
        pcd=pcd.voxel_down_sample(voxel_size=downsample)
        print("AFTER DOWNSAMPLING",len(pcd.points))

    

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)
def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(1.0, 0.0)
    return False

def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--csvPath",help="CSV Path for point cloud",required=True)
    parser.add_argument("--label",help="which label to show",default="no_background")
    parser.add_argument("--downsample",help="Voxel Downsampling",type=float,default=0)

    args=parser.parse_args()

    visualize_point_cloud(args.csvPath,args.label,args.downsample)



if __name__=="__main__":
    main()
