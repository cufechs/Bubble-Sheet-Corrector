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
from skimage.filters import median
from skimage.feature import canny
from skimage.measure import label
from skimage.color import label2rgb
from sklearn import cluster
from skimage.filters import threshold_otsu,apply_hysteresis_threshold
from skimage.morphology import closing,dilation


# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

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


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')


def perspective_correction(image):

    # load the image, convert it to grayscale, blur it
    # slightly, then find edges
    gray = np.uint8(rgb2gray(image)*256)
    ret,binary = cv2.threshold(gray,125,255,cv2.THRESH_BINARY)

    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if len(cnts)==2 else cnts[1]

    docCnt = None
    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in
        # descending order
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # loop over the sorted contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points,
            # then we can assume we have found the paper
            if len(approx) == 4:
                x,y = image.shape[:2]
                if peri > (x+y)*2/8: 
                    #checking if contour length is bigger than the original image's length/8
                    paper = four_point_transform(image, approx.reshape(4, 2))
                    return paper, True

    return None, False
