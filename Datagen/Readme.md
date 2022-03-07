This module deals with Point cloud generation from voxel images using canny detector

# Required libraries
` Numpy,
 Open3d,
 nibabel,
tqdm,
pandas,
itk,
pytorch`

# To generate point cloud from 3d Voxel image

### 1. Edge Detection
``` 
python3 canny.py --inputDir {Input Image Directory} --outputDir { Output Image Directory for Saving} --variance --lowerT {Lower Threshold (defaut 20)} --upperT {Upper Threshold(default 50)}
```
```
# For more Help
python3 canny.py -h
```

### 2. Point Cloud Generation

```
python3 pointcloud.py --edgeDir {Directory for Edges} --segDir {Segmentation Mask Directory} --outputDir {Directory to save Point Cloud CSV}
```

### 3. Visualize
```
python3 visualize.py --csvPath {Path for point cloud CSV}
```
