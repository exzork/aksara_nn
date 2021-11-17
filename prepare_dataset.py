import cv2
import numpy as np
import math
import scipy.ndimage as ndi
from skimage import morphology

dataset_dir = "./dataset/"

import os
labels = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]

for label in labels:
  for filename in os.listdir(dataset_dir+label):
    if not os.path.exists("./dataset_canny/"):
      os.makedirs(os.path.dirname("./dataset_canny/"))
    if not os.path.exists("./dataset_canny/"+label):
      os.makedirs("./dataset_canny/"+label)
    img_gray = cv2.imread(dataset_dir+label+"/"+filename,0)
    img_gauss = cv2.GaussianBlur(img_gray,(3,3),cv2.BORDER_DEFAULT)
    edge = cv2.Canny(img_gauss,100,200)
    cv2.imwrite("dataset_canny/"+label+"/"+filename,edge)

def preprocess(label,filename):
  img1 = cv2.imread("dataset_canny/"+label+"/"+filename,0)
  h,w = img1.shape

  x1=w-1
  y1=h-1
  x2=y2=0

  for y in range(0,h-1):
    for x in range(0,w-1):
      if img1[y,x]==255:
        if x>=x2:
          x2=x
        elif y>=y2:
          y2=y

  for y in range(h-1,0,-1):
    for x in range(w-1,0,-1):
      if img1[y,x]==255:
        if x<=x1:
          x1=x
        elif y<=y1:
          y1=y

  x2+=1
  if (y2-y1)%2 == 1:
    y2+=1
  if (x2-x1)%2 == 1:
    x2+=1 

  img_cropped = img1[y1:y2,x1:x2]
  hc,wc = img_cropped.shape

  s = abs(hc-wc)

  img2 = img_cropped
  if wc>hc:
    img2 = cv2.copyMakeBorder(img_cropped,round(s/2),round(s/2),0,0,cv2.BORDER_CONSTANT)
  elif wc<hc:
    img2 = cv2.copyMakeBorder(img_cropped,0,0,round(s/2),round(s/2),cv2.BORDER_CONSTANT)
  
  max_unit = max(hc,wc)
  dilate_by = math.ceil(max_unit/30)+2
  kernel_dilate = np.ones((dilate_by, dilate_by), 'uint8')
  
  img2 = cv2.dilate(img2,kernel_dilate,iterations=1)
  img2 = cv2.resize(img2,(30,30),cv2.INTER_AREA)
  img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)[1]

  if not os.path.exists("./dataset_30x30/"):
      os.makedirs(os.path.dirname("./dataset_30x30/"))
  if not os.path.exists("./dataset_30x30/"+label):
      os.makedirs("./dataset_30x30/"+label)
  cv2.imwrite("./dataset_30x30/"+label+"/"+filename,img2)
for label in labels:
  for filename in os.listdir(dataset_dir+label):
    preprocess(label,filename)