Python codes for generating point clouds

### 1. Libraries needed

1. numpy ```pip install numpy```
2. nibabel ```pip install nibabel```
3. pandas ```pip install pandas```
4. open3d ```pip install open3d```. [Click here](http://www.open3d.org/) for more.

### 2. To Run

To create point clouds from visceral dataset

``` python3 pointcloud.py --imageDir {Image Directory} --segDir {Segmentation Mask Directory}```

For help

``` python3 pointcloud.py -h ```

### 3. To Visualize after point cloud generation

``` python3 visualize.py --csvPath {Csv Path} ```
