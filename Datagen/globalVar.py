import numpy as np


#! FOR VISCERAL DATASET
radLexIDDict={1247:"trachea",1302:"right lung",1326:"left lung",170:" pancreas",
187:"gallbladder",237:"urinary bladder",2473: "sternum",29193:"firdt lumber vertebra",
29662:"right kidney",29663:"left kidney",30324:"right adrenal gland",
30325:"left adrenal gland",32248:"right psoas major",32249:"left psoas major",
40357:"right rectus abdominis",40358: "left rectus abdominis",480:"aorta",
58: "liver",7578:"thyroid gland",86:"spleen"}

ORGAN_CHOICE={"liver":58,"right_lung":1302,"left_lung":1326,
"bladder":237,"left_kidney":29663,"right_kidney":29662}

# ! FOR CT ORG DATASET
# Before distinguishing the two kidneys
CTORG_LABEL_TO_ORGAN={0: "background",
1: "liver",
2: "bladder",
3: "lungs",
4: "kidneys",
5: "bone",
6: "brain"}

# After distinguishing the two kidneys
CTORG_ORGAN_CHOICE={"background":0,
"liver":1,
"bladder":2,
"lungs":3,
"left_kidney":4,
"right_kidney":7}


# ! GLOBAL VARIABLES
ORGAN_TO_LABEL={"background":0,"liver":1,"lungs":2,
"bladder":3,"left_kidney":4,"right_kidney":5}

LABEL_TO_ORGAN={0:"background",1:"liver",2:"lungs",
3:"bladder",4:"left_kidney",5:"right_kidney"}

ORGAN_TO_RGB={"background":np.array([130,  15, 178])/255.0, "liver":np.array([232, 72, 80])/255.0,
"lungs":np.array([190, 254, 249])/255.0,"bladder":np.array([27, 7, 65])/255.0,
"left_kidney":np.array([79, 116, 172])/255.0,"right_kidney":np.array([79, 152, 172])/255.0}
