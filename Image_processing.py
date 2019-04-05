
import tensorflow as tf # import as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math
from scipy import ndimage



def deskew_line(image, thresh):

# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
  coords = np.column_stack(np.where(thresh > 0))
  angle = cv2.minAreaRect(coords)[-1]

  # the `cv2.minAreaRect` function returns values in the
  # range [-90, 0); as the rectangle rotates clockwise the
  # returned angle trends to 0 -- in this special case we
  # need to add 90 degrees to the angle
  if angle < -45:
    angle = -(90 + angle)

  # otherwise, just take the inverse of the angle to make
  # it positive
  else:
    angle = -angle
  print(angle)

  # rotate the image to deskew it
  (h, w) = image.shape[:2]
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated = cv2.warpAffine(image, M, (w, h),
    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
  return rotated

def predict_digit(predict_data):
    model_path = '/content/drive/My Drive/ml_models/mnist_digit.model' #my model
#     model_path = '/content/drive/My Drive/ml_models/digit_classifier.h5'
    # global ml_digit_model
    # predict_data = preprocess_image()
    predict_data_norm = tf.keras.utils.normalize(predict_data, axis=1) #normalized image array for prediction
    predict_data_numpy = np.array(predict_data_norm).reshape(-1, 28, 28, 1) # nparray
    # print(predict_data_norm)
#     ml_digit_model = tf.keras.models.load_model(model_path)
    ml_digit_model = model
    predictions = ml_digit_model.predict(predict_data_numpy)
    pred = predictions[0]
    # print(apps.ml_digit_model.summary())
    # for i, pred in enumerate(predictions):
      # print(np.argmax(pred))
      # print(np.amax(pred))
    if np.amax(pred) > 0.88:
        # print('good')
      # else:
      #     print("bad prediction")
      # print(pred)
        return str(np.argmax(pred))
    else:
        return "none"

def image_refiner(gray):
    org_size = 22
    img_size = 28
    rows,cols = gray.shape

    if rows > cols:
        factor = org_size/rows
        rows = org_size
        cols = int(round(cols*factor))
    else:
        factor = org_size/cols
        cols = org_size
        rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows))

    #get padding
    colsPadding = (int(math.ceil((img_size-cols)/2.0)),int(math.floor((img_size-cols)/2.0)))
    rowsPadding = (int(math.ceil((img_size-rows)/2.0)),int(math.floor((img_size-rows)/2.0)))

    #apply apdding
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    return gray

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
    number = ""
    path = 'sc.png'
    img_org = cv2.imread(path,2)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    ret,thresh = cv2.threshold(img,127,255,0)
#     edged = cv2.Canny(img, 75, 255)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     Sort contours left->right and up->down
#     contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * img.shape[1] )
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])#left->right
    #index of heirarchy for corresponding sorted contours
    h_index = sorted_hierarchy_index(contours, sorted_contours)
    for j,cnt in enumerate(sorted_contours):
        this_h_index = h_index[j]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR);
        cv2.drawContours(img_rgb, [cnt], 0, (0,255,0), 1) # last number is the thickness of line
        epsilon = 0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        hull = cv2.convexHull(cnt)
        k = cv2.isContourConvex(cnt)
        x,y,w,h = cv2.boundingRect(cnt)

#         if(hierarchy[0][j][2]!=1 and w>10 and h>10):

        # if hierarchy[0][this_h_index][3] == 0:
        if w>5 and h > 5:
        #putting boundary on each digit
            cv2.rectangle(img_org,(x,y),(x+w,y+h),(0,255,0),2)

            #cropping each image and process
            roi = img[y:y+h, x:x+w]
            roi = cv2.bitwise_not(roi)
            # roi = image_refiner(roi)
            th,fnl = cv2.threshold(roi,127,255,cv2.THRESH_BINARY)
            #preprocess as per MNIST standards
            # roi = remove_empty_pix(roi)
            plt.imshow(roi)
            plt.show()

multi_digit_preprocess()


