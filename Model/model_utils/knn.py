import pynanoflann
import torch
import numpy as np

def findKNN(points,queries,num_neighbors,radius=20):
    points=points.cpu().contiguous().numpy()
    queries=queries.cpu().contiguous().numpy()

    idx,dist=[],[]

    B=points.shape[0]

    for i in range(B):
        nn=pynanoflann.KDTree(n_neighbors=num_neighbors,metric="L2",radius=radius)
        nn.fit(points[i])
        tempDIST,tempIDX=nn.kneighbors(queries[i])
        idx.append(tempIDX.astype(np.int64))
        dist.append(tempDIST.astype(np.float32))
    
    idx=torch.from_numpy(np.array(idx))
    dist=torch.from_numpy(np.array(dist))
    return idx,dist


