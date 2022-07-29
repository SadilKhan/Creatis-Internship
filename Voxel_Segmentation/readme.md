# Segment the voxels in the visceral dataset.

```
python3 vsknn.py
--modelPath {.pth file - Path to the trained RandLaNet Model}
--csvPath {CSV Path for a point cloud which will be used in the trained model for prediction}
--imagePath {The visceral image from where the point cloud was generated}
--organ {Name of the organ to segment. Only binary segmentation can be performed in this version.}
--segPath1 { Segmentation Mask 1}
--segPath2 { Segmentation Mask 2. This is only for lungs since right lungs mask and left lungs mask are merged for single mask. Use "No" for other organs}
--neighbor {Number of neighbors in KNN interpolation (default 8)}
--outputPath {Output Path for the prediction label}
```
