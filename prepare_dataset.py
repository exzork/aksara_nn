import re
import cv2
import numpy as np
import math
import os
import scipy as sp
import scipy.ndimage

class prepare_dataset:

  @staticmethod
  def preprocess(img):

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gauss = cv2.GaussianBlur(img_gray,(3,3),cv2.BORDER_DEFAULT)
    edge = cv2.Canny(img_gauss,100,200)
    img1 = edge
    

    h,w = img1.shape

    x1=w-1
    y1=h-1
    x2=y2=0

    for y in range(0,h-1):
      for x in range(0,w-1):
        if img1[y,x]==255:
          if x>=x2:
            x2=x
          if y>=y2:
            y2=y

    for y in range(h-1,0,-1):
      for x in range(w-1,0,-1):
        if img1[y,x]==255:
          if x<=x1:
            x1=x
          if y<=y1:
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
    dilate_by = math.ceil(max_unit/30)+1
    kernel_dilate = np.ones((dilate_by, dilate_by), 'uint8')
    
    img2 = cv2.dilate(img2,kernel_dilate,iterations=1)
    img2 = flood_fill(img2)

    img2 = cv2.resize(img2,(30,30), interpolation=cv2.INTER_AREA)
    img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)[1]
    
    return img2    

def flood_fill(test_array,h_max=255):
    input_array = np.copy(test_array) 
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array