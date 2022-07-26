import os
import shutil
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def move_file(source_path,destination_path):
   shutil.move(source_path,destination_path)


def main():
   parser=argparse.ArgumentParser()
   parser.add_argument("--sourceDir",type=str)
   parser.add_argument("--destinationDir",type=str)
   
   args=parser.parse_args()

   allFiles=os.listdir(args.sourceDir)
   train,test=train_test_split(allFiles,test_size=0.2,random_state=1)
   
   for file in tqdm(train):
      if (file!="train") and (file!="test"):
         move_file(args.sourceDir+f"/{file}",args.destinationDir+f"/train/{file}")
   for file in tqdm(test):
      if (file!="train") and (file!="test"):
         move_file(args.sourceDir+f"/{file}",args.destinationDir+f"/test/{file}")

if __name__=='__main__':
   main()
   

