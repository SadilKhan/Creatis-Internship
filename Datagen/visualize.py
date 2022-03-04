import open3d as o3d
import pandas as pd
import numpy as np
import argparse


def visualize_point_cloud(csvPath):
    """ 3d visualization of point clouds """

    pointCloud=pd.read_csv(csvPath)
    unique_label=pointCloud["label"].unique()
    labelDict=dict()
    colorDict=dict()
    # Make colors for every unique label
    for l in unique_label:
        colorDict[l]=np.random.choice(range(256), size=3)
        labelDict[l]=list(colorDict[l]/255.0)
    print("Color Code",colorDict)

    # Make color lists
    colors=np.array([labelDict[l] for l in pointCloud["label"]])

    # open3d visualization
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(pointCloud[["x","y","z"]].values)
    pcd.colors=o3d.utility.Vector3dVector(colors)  

    o3d.visualization.draw_geometries([pcd])


def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--csvPath",help="CSV Path for point cloud",required=True)

    args=parser.parse_args()

    visualize_point_cloud(args.csvPath)



if __name__=="__main__":
    main()
