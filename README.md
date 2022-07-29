This module deals with Point cloud generation from voxel images using canny detector

# Required libraries
` Numpy,
 Open3d,
 nibabel,
tqdm,
pandas,
itk,
pytorch
pynanoflann`

`itk` may not work in Mac M1, use `simpleITK` instead.

# Folders

**1.Datagen:** Codes for generating point cloud from voxel images.

**2.Model:** Randla Net Model codes and training.

**3.Voxel Segmentation:** Codes for segmenting voxels using the trained model.

**4.Misc:** Miscellaneous Codes.


# To Produce the results

```
1. Download the visceral/ctorg dataset
2. Generate the point cloud using the steps described in the Datagen folder.
3. Train the model using the steps described in Model folder.
4. Segment the whole 3D Image using the steps described in Voxel Segmentation folder.
```
