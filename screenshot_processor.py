import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math
from scipy import ndimage

def sorted_hierarchy_index(contours, sorted_contours):
  print(type(sorted_contours))
  index_list = []
  for cnt in sorted_contours:
    #contours is python list and cnt is numpy array
    #so convert to list becore comparning
    temp_list = [i for i, e in enumerate(contours) if e.tolist() == cnt.tolist()]
    index_list+=temp_list
    print(index_list)
  return index_list

def multi_digit_preprocess():
    path = 'sc.png'
    img_org = cv2.imread(path,2)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    ret,thresh = cv2.threshold(img,127,255,0)
#     edged = cv2.Canny(img, 75, 255)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[4]
    cv2.drawContours(img_org, [cnt], 0, (0,255,0), 3)
    plt.imshow(img_org)
    plt.show()
def detect_circles():
    img = cv2.imread('sc.png',0)
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
    # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    plt.imshow(cimg)
    plt.show()

if __name__ == "__main__":
    multi_digit_preprocess()
    # detect_circles()
