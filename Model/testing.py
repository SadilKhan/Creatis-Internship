import torch
import pandas as pd
import argparse
def printHello(i,x,y,z):
 print(x+y+z+i)

if __name__=="__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument("--x",type=int,default=2)
  parser.add_argument("--y",type=int,default=2)
  parser.add_argument("--z",type=int,default=2)
  parser.add_argument("--i",type=int,default=2)
  args=parser.parse_args()
  printHello(args.i,args.x,args.y,args.z)
