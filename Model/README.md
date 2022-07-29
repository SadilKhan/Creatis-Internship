# Train the RandLa Net Model

```
python3 train.py
--csvPath { Path to all the point cloud files}
--epochs { Number of epochs}
--dataset_type { Visceral or CtOrg}
--batch_size { Default 1}
--decimation { The ratio at which the point cloud is divided by at each layer}
--neighbors { Number of neighbors considered by k-NN}
--nsplit { Cross Validation Split. Default 5}
--name {Experiment Name}
--model_name {Name of the model for saving the model as .pth file}
```
