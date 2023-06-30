Code for my master thesis at [Creatis Lab, Insa Lyon](https://www.creatis.insa-lyon.fr/site7/en).

Full pdf is available [here](https://mdsadilkhan.onrender.com/publications/masterthesis/report.pdf)

# Architecture
![Screenshot from 2023-06-30 16-37-51](https://github.com/SadilKhan/Creatis-Internship/assets/45021067/bac0dac2-1e75-47bc-bde1-6b783978382a)

# Results

![Screenshot from 2023-06-30 16-37-03](https://github.com/SadilKhan/Creatis-Internship/assets/45021067/7de1201a-ec31-482c-8628-5970b212f00d)


![Screenshot from 2023-06-30 16-35-29](https://github.com/SadilKhan/Creatis-Internship/assets/45021067/a94ccbd1-83ad-4b7a-889d-1032001533c5)

![Screenshot from 2023-06-30 16-36-10](https://github.com/SadilKhan/Creatis-Internship/assets/45021067/0b8bc115-28c0-40ae-a1e4-60db32c4f4ab)


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
