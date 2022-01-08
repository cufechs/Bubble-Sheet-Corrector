import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv,gray2rgb,rgba2rgb
from skimage.draw import circle_perimeter
from skimage.transform import (hough_line, hough_line_peaks,hough_circle,hough_circle_peaks)
from skimage.transform import resize

import matplotlib.pyplot as plt
from matplotlib import cm
import math
import cv2

from imutils.perspective import four_point_transform
from imutils import contours
import imutils

from openpyxl import load_workbook

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median,sobel_h, sobel, sobel_v,roberts, prewitt
from skimage.feature import canny
from skimage.measure import label
from skimage.color import label2rgb
from sklearn import cluster
from skimage.filters import threshold_otsu,apply_hysteresis_threshold
from skimage.morphology import closing,dilation,thin

from mlp_train import detect_digit


def loadImage(path):
    image = io.imread(path)
    
    if len(image.shape) == 4:
        image = rgba2rgb(image)
    
    if len(image.shape) > 2:
        image = (rgb2gray(image)*255).astype('uint8')
    else:
        image = (image).astype('uint8')
    return image

# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()  


def perspective_correction(image):
    def findContours1():
        gray = np.uint8(rgb2gray(image.copy())*255)
        _,binary = cv2.threshold(gray,125,255,cv2.THRESH_BINARY)
        cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts)==2 else cnts[1]
        show_images([binary])
        return cnts

    def findContours2():
        gray = np.uint8(rgb2gray(image.copy())*255)
        edged = np.uint8(canny(gray,sigma=4))
        cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts)==2 else cnts[1]
        show_images([edged])
        return cnts
    
    
    getContours = [findContours1, findContours2]
    for Contours in getContours:
        cnts = Contours()
        
        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                # if our approximated contour has four points,
                # then we can assume we have found the paper
                print(len(approx))
                if len(approx) == 4:
                    x,y = image.shape[:2]
                    if peri > (x+y)*2/4: 
                        #checking if contour length is bigger than the original image's length/8
                        paper = four_point_transform(image, approx.reshape(4, 2))
                        return paper, True

    return image, False

def removeShadow(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result
    
def get_id(img, id_length=7, show_info=False):
    
    image,_ = perspective_correction(img.copy())
    image = image[:image.shape[0]//4,image.shape[1]//2:]
    image = removeShadow(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    _,binary = cv2.threshold(gray,235,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
    if show_info: print("Number of contours:" + str(len(contours)))
        
    data = []
    data_i = []
    for i,c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        #if w > image.shape[0]/8 or h > image.shape[1]/8: #if very large contour 
        #    continue
        if w/h<0.8 or w/h>1.2: # if rec is not a square
            continue
        if cv2.contourArea(c) < 300: # if the square is so small
            continue
            
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        data.append((y,y+h,x,x+w))

    if show_info: print("Number of contours:" + str(len(data)))

        
    id_str = id_length*'0'
    if len(data) >= id_length:
        # To take the low 7 squares in the image, to avoid the logo's detecction
        data.sort(key=lambda element: element[0], reverse=True)
        data = data[:id_length]
        data.sort(key=lambda element: element[2])
        ###

        for y1,y2,x1,x2 in data:
            data_i.append(thin(np.invert(binary[y1+2:y2-2,x1+2:x2-2].copy()),10))

        id_str = ''
        for i in range(id_length):
            p = detect_digit(data_i[i], plot=show_info)
            id_str += str(p)
        

    if show_info: 
        plt.gray()

        plt.figure(figsize=(15, 15))
        plt.imshow(gray)
        plt.title('gray')
        plt.figure(figsize=(15, 15))
        plt.imshow(binary)
        plt.title('binary')
        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.title('image');

        show_images(data_i)
        
    return id_str
    
