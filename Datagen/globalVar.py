import numpy as np



radLexIDDict={1247:"trachea",1302:"right lung",1326:"left lung",170:" pancreas",
187:"gallbladder",237:"urinary bladder",2473: "sternum",29193:"firdt lumber vertebra",
29662:"right kidney",29663:"left kidney",30324:"right adrenal gland",
30325:"left adrenal gland",32248:"right psoas major",32249:"left psoas major",
40357:"right rectus abdominis",40358: "left rectus abdominis",480:"aorta",
58: "liver",7578:"thyroid gland",86:"spleen"}

ORGAN_CHOICE={"liver":58,"pancreas":170,"spleen":86,"gallbladder":187,"urinary bladder":237,"left kidney":29663,"right kidney":29662}

ORGAN_TO_LABEL={"background":0,"liver":1,"pancreas":2,"spleen":3,"gallbladder":4,"urinary bladder":5,"left kidney":6,"right kidney":7}

ORGAN_TO_RGB={"background":np.array([130,  15, 178])/255.0, "liver":np.array([232, 72, 80])/255.0,
"pancreas":np.array([190, 254, 249])/255.0,"spleen":np.array([126,   3,  22])/255.0,
"gallbladder":np.array([124,  87, 238])/255.0,"urinary bladder":np.array([27, 7, 65])/255.0,
"left kidney":np.array([123, 225, 193])/255.0,"right kidney":np.array([ 77, 220,  34])/255.0}
