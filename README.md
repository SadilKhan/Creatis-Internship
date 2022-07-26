This module deals with Point cloud generation from voxel images using canny detector

# Required libraries
` Numpy,
 Open3d,
 nibabel,
tqdm,
pandas,
itk,
pytorch`

`itk` may not work in Mac M1, use `simpleITK` instead.

# To generate point cloud from 3d Voxel image

### (Optional) for CtOrg
The Ctorg dataset has same labels for right and left kidney, if you want to label them separately, then the run the following code

```
python3 ctOrgSeg.py 
--segDir (The segmentation Directory)
```

### 1. Edge Detection

For windows, Linux, Intel Macs
``` 
python3 canny.py 
--inputDir {Input Image Directory} 
--outputDir { Output Image Directory for Saving} 
--variance 
--lowerT {Lower Threshold (defaut 20)} 
--upperT {Upper Threshold(default 50)}
```
```
# For more Help
python3 canny.py -h
```

For Mac M1
``` 
python3 cannyv2.py 
--inputDir {Input Image Directory} 
--outputDir { Output Image Directory for Saving} 
--variance 
--lowerT {Lower Threshold (defaut 20)} 
--upperT {Upper Threshold(default 50)}
```
```
# For more Help
python3 cannyv2.py -h
```

### 2. Point Cloud Generation

```
python3 pointcloud.py 
--edgeDir {Directory for Edges} 
--segDir {Segmentation Mask Directory} 
--outputDir {Directory to save Point Cloud CSV}
```

### 3. Visualize
```
python3 visualize.py 
--csvPath {Path for point cloud CSV} 
--label {all,no_background,liver,spleen,urinary bladder,gallbladder,right kidney,left kdiney,pancreas}
--downsample {Voxel Downsampling (default 0)}
```
