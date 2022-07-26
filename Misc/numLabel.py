import argparse
import os
import pandas as pd
from glob import glob
import numpy as np
from tqdm import tqdm

def numLabel(path=""):
    data=pd.read_csv(path)
    print(f"CSV:"+path.split("/")[-1]+f"\t Labels:{data['label'].nunique()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvDir",type=str,default="/home/khan/Internship/dataset/Extracted/point_cloud")

    args=parser.parse_args()
    CSVLIST=glob(os.path.join(args.csvDir,"*.csv"))

    for csv in CSVLIST:
        numLabel(csv)

if __name__ == "__main__":
    main()
