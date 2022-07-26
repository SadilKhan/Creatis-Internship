
import os ,gc
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
def find_weights(path):
    allFiles=os.listdir(path)
    data=pd.read_csv(path+"/"+allFiles[0])
    value=data['label'].value_counts()
    value=dict(value)
    del data
    gc.collect()
    for i in tqdm(range(1,len(allFiles))):
        data=pd.read_csv(path+"/"+allFiles[i])
        temp=dict(data['label'].value_counts())
        for k,v in temp.items():
           value[k]+=v
        #value+=data['label'].value_counts()
        del data,temp
        gc.collect()
    for k,v in value.items():
        value[k]/=len(allFiles)
    print(value)

if __name__ == "__main__":
    find_weights("/home/khan/Internship/dataset_res/Extracted/point_cloud/train")

