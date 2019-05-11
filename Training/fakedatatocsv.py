
import pandas as pd
import numpy as np
import glob
import cv2
import os
import random


#inputPath = "/home/kc/Downloads/keras-multi-input/get_files/Training/Wastes Dataset/WastesInfo.txt"
#cols = ["weight", "top_area", "front_area", "time", "category"]
#df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
#df.pop('category')


wgt=[]

for i in range(410):
   wgt.append(random.randrange(20, 40))

for i in range(403):
   wgt.append(random.randrange(10, 30))

for i in range(501):
   wgt.append(random.randrange(100, 200))

for i in range(594):
   wgt.append(random.randrange(0, 20))

for i in range(482):
   wgt.append(random.randrange(15, 40))

for i in range(189):
   wgt.append(random.randrange(0, 300))



top=[]

for i in range(410):
   top.append(random.randrange(0, 100))

for i in range(403):
   top.append(random.randrange(0, 400))

for i in range(501):
   top.append(random.randrange(0, 150))

for i in range(594):
   top.append(random.randrange(0, 100))

for i in range(482):
   top.append(random.randrange(0, 200))

for i in range(189):
   top.append(random.randrange(0, 600))


frt=[]

for i in range(410):
   frt.append(random.randrange(0, 100))

for i in range(403):
   frt.append(random.randrange(0, 50))

for i in range(501):
   frt.append(random.randrange(0, 150))

for i in range(594):
   frt.append(random.randrange(0, 50))

for i in range(482):
   frt.append(random.randrange(0, 200))

for i in range(189):
   frt.append(random.randrange(0, 600))


tim=[]

for i in range(410):
   tim.append(random.randrange(11, 17))

for i in range(403):
   tim.append(random.randrange(14, 18))

for i in range(501):
   tim.append(random.randrange(7, 18))

for i in range(594):
   tim.append(random.randrange(14, 18))

for i in range(482):
   tim.append(random.randrange(7, 18))

for i in range(189):
   tim.append(random.randrange(12, 15))


cat =[]

for i in range(410):
   cat.append('aluminium')

for i in range(403):
   cat.append('cardboard')

for i in range(501):
   cat.append('glass')

for i in range(594):
   cat.append('paper')

for i in range(482):
   cat.append('plastic')

for i in range(189):
   cat.append('trash')

#d = {'weight' : pd.Series(wgt), 'top_area' :pd.Series(top), 'front_area' :pd.Series(frt), 'time' :pd.Series(tim), 'category' :pd.Series(cat)}  

d = {'weight' : wgt, 'top_area' : top, 'front_area' :frt, 'time' :tim, 'category' :cat}

df = pd.DataFrame(d)
df = df[['weight','top_area','front_area','time','category']]



#pd: your pandas dataframe

#filedir = "/home/kc/Downloads/keras-multi-input/get_files/Training/Wastes Dataset/WastesInfo.txt"
#with open(filedir,'w') as outfile:
#    pd.to_string(outfile)
#Neatly allocate all columns and rows to a .txt file

df.to_csv(r'/home/kc/Downloads/keras-multi-input/get_files/Training/Wastes Dataset/WastesInfo.csv', header=None, index=None, sep=' ', mode='a')
#print df


