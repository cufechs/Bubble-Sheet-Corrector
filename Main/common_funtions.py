import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import canny
from skimage.morphology import thin
from skimage.transform import resize
from scipy.signal import convolve2d

from imutils.perspective import four_point_transform

from mlp_train import detect_digit

import cv2

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
        return cnts

    def findContours2():
        gray = np.uint8(rgb2gray(image.copy())*255)
        edged = np.uint8(canny(gray,sigma=4))
        cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts)==2 else cnts[1]
        return cnts
    
    getContours = [findContours2, findContours1]
    for Contours in getContours:
        cnts = Contours()
        
        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                # if our approximated contour has four points,
                # then we can assume we have found the paper
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
    
def cropDigit(img,padding=3):
    im = img.copy()
    im = im > 100/255
    y,x = im.shape
    
    flag = False
    for i in range(y):
        for j in range(x):
            if im[i][j]!=0:
                flag=True; break
        if flag: break
    y_upper = i
    
    flag = False
    for i in range(y-1,0,-1):
        for j in range(x):
            if im[i][j]!=0:
                flag=True; break
        if flag: break
    y_lower = y-i
    
    flag = False
    for j in range(x):
        for i in range(y):
            if im[i][j]!=0:
                flag=True; break
        if flag: break
    x_left = j
    
    flag = False
    for j in range(x-1,0,-1):
        for i in range(y):
            if im[i][j]!=0:
                flag=True; break
        if flag: break
    x_right = x-j
        
    maxy,miny = max(y_upper,y_lower), min(y_upper,y_lower)
    maxy_i = 0 if y_upper>y_lower else 1
    maxx,minx = max(x_left,x_right), min(x_left,x_right)
    maxx_i = 0 if x_left>x_right else 1
    
    im = img.copy()
    
    shift_right = np.array([
    [ 0, 0, 0],
    [ 0, 0, 1],
    [ 0, 0, 0]
    ])
    shift_left = np.array([
    [ 0, 0, 0],
    [ 1, 0, 0],
    [ 0, 0, 0]
    ])
    
    if x_left > x_right:
        for _ in range((x_left-x_right)//2): 
            im = convolve2d(im, shift_left)
            im = im[1:-1,1:-1]
            x_left-=1
            x_right+=1
    elif x_right > x_left:
        for _ in range((x_right-x_left)//2): 
            im = convolve2d(im, shift_right)
            im = im[1:-1,1:-1]
            x_left+=1
            x_right-=1
                
    x_left = (maxy+miny)//2
    x_right = (maxy+miny)//2
            
    y_upper = y_upper - padding if y_upper>padding else 0
    y_lower = y_lower - padding if y_lower>padding else 1
    x_left = x_left - padding if x_left>padding else 0
    x_right = x_right - padding if x_right>padding else 1
        
    return resize(im[y_upper:-y_lower,x_left:-x_right],(28,28)) 


def get_id(img, id_length=7, show_info=False, correct_perspective = False):
    
    if correct_perspective:
        image,_ = perspective_correction(img.copy())
    else: 
        image = img.copy()
        
    image = image[image.shape[0]//16:image.shape[0]//4,image.shape[1]//2:]
    image = removeShadow(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    _,binary = cv2.threshold(gray,235,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
    if show_info: print("Number of contours:" + str(len(contours)))
        
    data = []
    data_cells=[]
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
        # To take the bigger 7 squares in the image, to avoid the logo's detection and digits
        data.sort(key=lambda x: (x[1]-x[0])*(x[3]-x[2]), reverse=True)
        data = data[:id_length]
        data.sort(key=lambda x: x[2]) #sorting rectangles from left to right
        ###
  
        for i,(y1,y2,x1,x2) in enumerate(data):
            im = resize(np.invert(binary[y1:y2,x1:x2].copy()),(32,32))
            im = im[2:-2,2:-2]
            data_cells.append(cropDigit(im,padding=2))

        id_str = ''
        for i in range(id_length):
            p = detect_digit(data_cells[i], plot=False)
            id_str += str(p)

    if show_info: 
        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.title('image');

        show_images(data_cells)
        
    return id_str