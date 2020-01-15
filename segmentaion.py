import sys
import os
import numpy as np
import pandas as pd
import cv2
import scipy.misc
from matplotlib import pyplot as plt
from pathlib import Path
from scipy import ndimage
from PIL import Image


def get_mask(img):

    # turn image to 2d-array and resize to right shape
    if len(img.shape) > 2:
        img = img[:,:,0]
    img = cv2.resize(img, (512,512))

    def eraseMax(img,eraseLineCenter=0,eraseLineWidth=30,draw=False):
        img_ = img.copy()
        sumpix0=np.sum(img_,0)
        if draw:
            plt.plot(sumpix0)
            plt.title('Sum along axis=0')
            plt.xlabel('Column number')
            plt.ylabel('Sum of column')
        max_r2=np.int_(len(sumpix0)/3)+np.argmax(sumpix0[np.int_(len(sumpix0)/3):np.int_(len(sumpix0)*2/3)])
        cv2.line(img_,(max_r2+eraseLineCenter,0),(max_r2+eraseLineCenter,512),0,eraseLineWidth)
        return img_

    img_erased = eraseMax(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_erased)   
  
    ker = 169
    kernel = np.ones((ker,ker),np.uint8)
    blackhat = cv2.morphologyEx(img_clahe, cv2.MORPH_BLACKHAT, kernel)    
  
    threshold = 45
    ret, thresh = cv2.threshold(blackhat, threshold, 255, 0)

    def get_cmask(img, maxCorners=3800, qualityLevel=0.001, minDistance=1,Cradius=6):
        corners = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance)
        corners = np.int0(corners)
        cmask = np.zeros(img.shape)
        for corner in corners:
            x,y = corner.ravel()
            cv2.circle(cmask,(x,y),Cradius,1,-1)
        return cmask

    cmask = get_cmask(img_clahe)
    mask = np.multiply(cmask,thresh).astype('uint8')
    median = cv2.medianBlur(mask,23)

    def contourMask(image):
        contours,hierc = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        area = np.zeros(len(contours))
        for j in range(len(contours)):
            cnt = contours[j]
            area[j] = cv2.contourArea(cnt)
        mask = np.zeros(image.shape)
        cv2.drawContours(mask, contours, np.argmax(area), (255), -1)#draw largest contour-usually right lung   
        temp = np.copy(area[np.argmax(area)])
        area[np.argmax(area)]=0
        if area[np.argmax(area)] > temp/10:#make sure 2nd largest contour is also lung, not 2 lungs connected
            cv2.drawContours(mask, contours, np.argmax(area), (255), -1)#draw second largest contour  
        contours.clear() 
        return mask

    contour_mask = contourMask(median).astype('uint8')
    return contour_mask

def segment(img, contour_mask):
    
    iw, ih = img.shape
    mask_resized = cv2.resize(contour_mask, (iw, ih))
    mask_resized = mask_resized/255
    slice_y, slice_x = ndimage.find_objects(mask_resized, 1)[0]
    h, w = slice_y.stop - slice_y.start, slice_x.stop - slice_x.start

    nw, nh = w, h
    dw, dh = (nw-w)//2, (nh-h)//2
    t = max(slice_y.start-dh-30, 0)
    l = max(slice_x.start-dw-50, 0)
    b = min(slice_y.stop+dh+10, ih)
    r = min(slice_x.stop+dw+50, iw)

    img_pil = Image.fromarray(img)
    cropped_image = img_pil.crop((l, t, r, b))
    numpy_cropped_image = np.array(cropped_image) 

    return numpy_cropped_image
