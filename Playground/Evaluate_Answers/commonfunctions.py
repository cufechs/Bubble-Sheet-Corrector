import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv,gray2rgb,rgba2rgb
from skimage.draw import circle_perimeter
from skimage.transform import (hough_line, hough_line_peaks,hough_circle,hough_circle_peaks)
from skimage.transform import resize
from skimage.morphology import binary_closing

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
        return cnts

    def findContours2():
        gray = np.uint8(rgb2gray(image.copy())*255)
        edged = np.uint8(canny(gray,sigma=4))
        cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts)==2 else cnts[1]
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
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result

def removeShadowGray(img):

    dilated_img = cv2.dilate(img, np.ones((7,7), np.uint8)) 
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = diff_img.copy() # Needed for 3.x compatibility
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return diff_img
    
    
def cropDigit(img,padding=3):
    im = img.copy()
    im = im > 150/255
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
    y_lower = y_lower - padding if y_lower>padding else 0
    x_left = x_left - padding if x_left>padding else 0
    x_right = x_right - padding if x_right>padding else 0
    
    if y_lower!=0 and x_right!=0:
        im = resize(im[y_upper:-y_lower,x_left:-x_right],(28,28)) 
    elif y_lower!=0 and x_right==0:
        im = resize(im[y_upper:-y_lower,x_left:],(28,28)) 
    elif y_lower==0 and x_right!=0:
        im = resize(im[y_upper:,x_left:-x_right],(28,28)) 
    else:
        im = resize(im[y_upper:,x_left:],(28,28)) 
        
    im_temp = im.copy()
    im_temp = im_temp > 50/255
        
    flag = False
    for i in range(y):
        for j in range(x):
            if im_temp[i][j]!=0:
                flag=True; break
        if flag: break
    y_upper = i
    
    flag = False
    for i in range(y-1,0,-1):
        for j in range(x):
            if im_temp[i][j]!=0:
                flag=True; break
        if flag: break
    y_lower = y-i
    
    shift_up = np.array([
    [ 0, 0, 0],
    [ 0, 0, 0],
    [ 0, 1, 0]
    ])
    shift_down = np.array([
    [ 0, 1, 0],
    [ 0, 0, 0],
    [ 0, 0, 0]
    ])
    
    if y_upper > y_lower:
        for _ in range((y_upper-y_lower)//2): 
            im = convolve2d(im, shift_down)
            im = im[1:-1,1:-1]
            
    elif y_lower > y_upper:
        for _ in range((y_lower-y_upper)//2): 
            im = convolve2d(im, shift_up)
            im = im[1:-1,1:-1]
            
    return im


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
            data_cells.append(cropDigit(im,padding=4))

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
    

def Hough(image):
    img = canny(image)
    (M,N) = img.shape
    R_max = 30 #np.max((M,N))
    R_min = 20
    threshold = 10
    region = 10
    R = R_max - R_min
    #Initializing accumulator array.
    #Accumulator array is a 3 dimensional array with the dimensions representing
    #the radius, X coordinate and Y coordinate resectively.
    #Also appending a padding of 2 times R_max to overcome the problems of overflow
    A = np.zeros((R_max,M+2*R_max,N+2*R_max))
    B = np.zeros((R_max,M+2*R_max,N+2*R_max))
    #Precomputing all angles to increase the speed of the algorithm
    theta = np.arange(0,360)*np.pi/180
    edges = np.argwhere(img[:,:]) #Extracting all edge coordinates
    for val in range(R):
        r = R_min+val
        #Creating a Circle Blueprint
        bprint = np.zeros((2*(r+1) - 1,2*(r+1) - 1))
        (m,n) = (r,r) #Finding out the center of the blueprint
        for angle in theta:
            x = int(np.round(r*np.cos(angle)))
            y = int(np.round(r*np.sin(angle)))
            bprint[m+x,n+y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x,y in edges: #For each edge coordinates
            #Centering the blueprint circle over the edges
            #and updating the accumulator array
            X = [x-m-1+R_max,x+m+R_max] #Computing the extreme X values
            Y = [y-n-1+R_max,y+n+R_max] #Computing the extreme Y values
            A[r,X[0]:X[1],Y[0]:Y[1]] += bprint
        A[r][A[r]<threshold*constant/r] = 0
    for r,x,y in np.argwhere(A):
        temp = A[r-region:r+region,x-region:x+region,y-region:y+region]
        try:
            p,a,b = np.unravel_index(np.argmax(temp),temp.shape)
        except:
            continue
        B[r+(p-region),x+(a-region),y+(b-region)] = 1
    circles = B[:,R_max:-R_max,R_max:-R_max] #removing padding
    return circles


def show_Hough(image,circles, showInfo=False):
    if showInfo: 
        fig = plt.figure()
        fig, ax = plt.subplots()
        plt.imshow(image)
    circleCoordinates = np.argwhere(circles) #Extracting the circle information
    if showInfo: 
        circle = []
        for r,x,y in circleCoordinates:
            circle.append(plt.Circle((y,x),r,color=(1,0,0),fill=False,linewidth=1))
            ax.add_patch(circle[-1])
        plt.show()
        print(len(circle))
    return circleCoordinates


def crop_answers_section(bubbles_w_cross, show_info=False):
    cross_template = loadImage("digital_images/bubble_w_cross.png")

    result=cv2.matchTemplate(bubbles_w_cross, cross_template, cv2.TM_CCORR_NORMED) 
    objects_matched = []   # get the highest 2 matches with templates and their locations (maxLoc)
    while len(objects_matched) < 2:         # 2 matches

        minV, maxV, minLoc, maxLoc = cv2.minMaxLoc(result)   

        #print(result[maxLoc[1]-3:maxLoc[1]+3,maxLoc[0]-3:maxLoc[0]+3]) 
        if show_info: print(maxLoc)

        for i in range(-3, 3):          # we remove the highest match to avoid re-matching again
            for j in range(-3, 3):  
                result[maxLoc[1] + i, maxLoc[0] + j] = 0
        
        # print(result[maxLoc[1]-3:maxLoc[1]+3,maxLoc[0]-3:maxLoc[0]+3])

        #get the centers of the highest match template
        maxLoc = list(maxLoc)  #tuple is immutable , convert to list
        maxLoc[0] = maxLoc[0] + cross_template.shape[1] // 2  
        maxLoc[1] = maxLoc[1] + cross_template.shape[0] // 2
        maxLoc = tuple(maxLoc)

        #add location of the match in objects matched
        objects_matched.append(maxLoc)

        objects_matched = sorted(objects_matched) 

    p1 = objects_matched[0]
    p2 = objects_matched[-1]
    bubbles_w_cross = bubbles_w_cross[p1[1]:p2[1], p1[0]:p2[0]]
    return bubbles_w_cross

def count_white(r,x,y,diff):
    top_l = (x-r//2,y-r//2)
    bottom_r = (x+r//2,y+r//2)
    count_white=0
    for ix in range(top_l[0],bottom_r[0]):
        for jy in range(top_l[1],bottom_r[1]):
            if diff[ix,jy]==1:
                count_white+=1
    return count_white

def loadModelAnswer(fileName):
    modelAnwser = []
    # open file with fileName and read the data:
    with open(fileName) as file:
        lines = file.readlines()
        for line in lines:
            data = line.rstrip()
            if data == "A" or data == "a" or data == "1":
                modelAnwser.append(1)
            elif data == "B" or data == "b" or data == "2":
                modelAnwser.append(2)
            elif data == "C" or data == "c" or data == "3":
                modelAnwser.append(3)
            elif data == "D" or data == "d" or data == "4":
                modelAnwser.append(4)
            elif data == "E" or data == "e" or data == "5":
                modelAnwser.append(5)
            elif data == "F" or data == "f" or data == "6":
                modelAnwser.append(6)
    
    return modelAnwser

def getAnswers(circleCoordinates,diff,show_info=False):
    # thresh = threshold_otsu(diff)
    # diff=diff > thresh
    #answers_closing = closing(diff,np.ones((3,3),dtype=int))
    #show_images([diff],["diff"])
    radii= circleCoordinates[:,0]
    radius=np.average(radii)
    #print(radius)
    ###
    #sort with x
    circleCoordinates[:,1] = np.sort(circleCoordinates[:,1])
    #print(circleCoordinates)
    # then categorize every answers with the same (near) x value ; diff between each x value is less than radius
    rows = []
    for r,x,y in circleCoordinates:
        if len(rows)==0:
            rows.append([(r,x,y)]) 
        elif np.abs(rows[-1][0][1] - x) < radius:
            rows[-1].append((r,x,y))
        else:
            rows.append([(r,x,y)]) 
    rows = np.array(rows)
    if show_info: print(rows)
    # sort by Y and see if any centers are repeated due to Hough errors!!
    ###
    needsModificationDueYindex = np.zeros(rows.shape[0],dtype=int)
    gotGoodY = False
    goodYindecies = []
    for i in range(rows.shape[0]):
        # sort each row with y
        rows[i,:,2] = np.sort(rows[i,:,2])
        #print("row =",rows[i,:,2]) 
        prevYindex = 0
        for yindex in range(1,len(rows[i,:,2])):
            if np.abs(rows[i,:,2][prevYindex] - rows[i,:,2][yindex]) < radius:
                needsModificationDueYindex[i] = 1
                #print('This row '+ str(i)+' needs modification!!')
                break
            prevYindex = yindex
        if needsModificationDueYindex[i] == 0 and not gotGoodY:
            goodYindecies = rows[i,:,2]
            #print(goodYindecies)
            gotGoodY = True
     ###
    cols = []
    foundCols = False
    if len(goodYindecies) == 0: 
        for i in range(rows.shape[0]):
            if foundCols:
                break
            # sort each row with y
            rows[i,:,2] = np.sort(rows[i,:,2])
            for yindex in range(1,len(rows[i,:,2])):
                if len(cols)==5:
                    foundCols = True
                    break
                insertCols = False
                if len(cols) == 0:
                    insertCols = True
                for yVal in cols:
                    insertCols = True
                    if np.abs(yVal - rows[i,:,2][yindex]) < radius:
                        # we found similar y before
                        insertCols = False
                        break
                if insertCols:
                    cols.append(rows[i,:,2][yindex])
    
    cols = sorted(cols)
    if show_info: print(cols)
    for i in range(len(needsModificationDueYindex)):
        if needsModificationDueYindex[i] == 1:
            if len(goodYindecies) == 0:
                rows[i,:,2] = cols
            else:
                rows[i,:,2] = goodYindecies
    #print(rows)
    ###
    # calculate answers and compare it with model answer:
    currentAnswer = np.zeros(rows.shape[0],dtype=int)
    perim_circ= np.pi * radius * 2          
    for rindex in range(rows.shape[0]):
        ansNo = 1
        for r,x,y in rows[rindex]:
            white_count = count_white(r,x,y,diff)
            #print(white_count,x,y)
            if white_count > perim_circ + 25: # 25 scaling issues i.e. saftey margin
                if currentAnswer[rindex] == 0:
                    #print('1st ans: at x,y , white_count -> perim_circ ', x ,y , white_count , perim_circ)
                    currentAnswer[rindex] = ansNo
                else:
                    #print('other ans: at x,y ,white_count -> perim_circ ', x ,y , white_count , perim_circ)
                    currentAnswer[rindex] = -1
            ansNo += 1
        #print('Next row')
    #print(currentAnswer)
    return currentAnswer
# def getAnswers(circleCoordinates,diff,show_info=False):
#     # get radii:
#     radii= circleCoordinates[:,0]
#     radius=np.average(radii)
#     #print(radius)
#     ###
#     #sort with x
#     circleCoordinates[:,1] = np.sort(circleCoordinates[:,1])
#     #print(circleCoordinates)
#     # then categorize every answers with the same (near) x value ; diff between each x value is less than radius
#     rows = []
#     for r,x,y in circleCoordinates:
#         if len(rows)==0:
#             rows.append([(r,x,y)]) 
#         elif np.abs(rows[-1][0][1] - x) < radius:
#             rows[-1].append((r,x,y))
#         else:
#             rows.append([(r,x,y)]) 
#     rows = np.array(rows)
#     cols = []#np.zeros(rows[0].shape[0])
#     # print(cols)
#     #if show_info: print(rows)
#     # sort by Y and see if any centers are repeated due to Hough errors!!
#     ###
#     needsModificationDueYindex = np.zeros(rows.shape[0],dtype=int)
#     foundCols = False
#     gotGoodY = False
#     goodYindecies = []
#     for i in range(rows.shape[0]):
#         # sort each row with y
#         rows[i,:,2] = np.sort(rows[i,:,2])
#         #print("row =",rows[i,:,2]) 
#         prevYindex = 0
#         for yindex in range(1,len(rows[i,:,2])):
#             if np.abs(rows[i,:,2][prevYindex] - rows[i,:,2][yindex]) < radius:
#                 needsModificationDueYindex[i] = 1
#                 #print('This row '+ str(i)+' needs modification!!')
#                 break
#             prevYindex = yindex
#         if needsModificationDueYindex[i] == 0 and not gotGoodY:
#             goodYindecies = rows[i,:,2]
#             #print(goodYindecies)
#             gotGoodY = True
# #             else:
# #                 # avg these goodYindecies
# #                 goodYindecies = (goodYindecies + rows[i,:,2])/2
#             #print(goodYindecies)
#             #gotGoodY = True
#     #print(needsModificationDueYindex)
#     if len(goodYindecies) == 0: 
#         for i in range(rows.shape[0]):
#             if foundCols:
#                 break
#             # sort each row with y
#             rows[i,:,2] = np.sort(rows[i,:,2])
#             for yindex in range(1,len(rows[i,:,2])):
#                 if len(cols)==5:
#                     foundCols = True
#                     break
#                 insertCols = False
#                 if len(cols) == 0:
#                     insertCols = True
#                 for yVal in cols:
#                     insertCols = True
#                     if np.abs(yVal - rows[i,:,2][yindex]) < radius:
#                         # we found similar y before
#                         insertCols = False
#                         break
#                 if insertCols:
#                     cols.append(rows[i,:,2][yindex])
    
#     cols = sorted(cols)
#     if show_info: print(cols)
#     for i in range(len(needsModificationDueYindex)):
#         if needsModificationDueYindex[i] == 1:
#             if len(goodYindecies) == 0:
#                 rows[i,:,2] = cols
#             else:
#                 rows[i,:,2] = goodYindecies
#     if show_info: print(rows)
#     ###
#     # calculate answers and compare it with model answer:
#     currentAnswer = np.zeros(rows.shape[0],dtype=int)
#     perim_circ= np.pi * radius * 2          
#     for rindex in range(rows.shape[0]):
#         ansNo = 1
#         for r,x,y in rows[rindex]:
#             white_count = count_white(r,x,y,diff)
#             #print(white_count,x,y)
#             if white_count > perim_circ:
#                 if currentAnswer[rindex] == 0:
#                     currentAnswer[rindex] = ansNo
#                 else:
#                     currentAnswer[rindex] = -1
#             ansNo += 1
#         #print('Next row')
#     #print(currentAnswer)
#     return currentAnswer


def getFinalAnswers(img, modelAnswerPath='Model_answer.txt', correctPerspective=False, showInfo=False):
    if showInfo: show_images([img])
    
    flag = True
    if correctPerspective:
        answers,flag = perspective_correction(img.copy())
    else:
        answers = img.copy()
    
    if showInfo: show_images([answers],["perspective"])

    paper_gray = rgb2gray(answers)*255
    paper_gray_resized = resize(paper_gray,(1600,1286))
    paper_gray_resized = paper_gray_resized.astype("uint8")

    paper_gray_resized = removeShadowGray(paper_gray_resized)
    out_ans = crop_answers_section(paper_gray_resized)
    
    bubbles_w_cross=loadImage("digital_images/bubbles_empty_with_cross.jpeg")
    out_ref = crop_answers_section(bubbles_w_cross)
    ref_shape = out_ref.shape
    
    ref_chunck1 =out_ref[:,0:370]  #seg bel 7ob
    ref_chunck2 =out_ref[:,450:]
    thresh1 = threshold_otsu(ref_chunck1)
    thresh2 = threshold_otsu(ref_chunck2)
    ref_chunck1 = ref_chunck1 > thresh1
    ref_chunck2 = ref_chunck2 > thresh2

    out_ans = resize(out_ans,ref_shape) 
    out_ans = removeShadowGray((out_ans*255).astype("uint8"))
    if showInfo: show_images([out_ans])

    ans_chunck1 =out_ans[:,0:370]  #seg bel 7ob
    ans_chunck2 =out_ans[:,450:]

    # resize chunks to match
    ans_chunck1 = resize(ans_chunck1,ref_chunck1.shape)
    ans_chunck2 = resize(ans_chunck2,ref_chunck2.shape)

    thresh1, thresh2 = 240/255, 240/255

    ans_chunck1 = (ans_chunck1 < thresh1)
    ans_chunck2 = (ans_chunck2 < thresh2)

    if showInfo: show_images([ref_chunck1,ref_chunck2])
    if showInfo: show_images([ans_chunck1,ans_chunck2])
    
    # closing bubbles to fill
    ans_chunck1 = binary_closing(ans_chunck1, np.ones((7,7), np.uint8))
    ans_chunck2 = binary_closing(ans_chunck2, np.ones((7,7), np.uint8))
    
    if showInfo: show_images([ans_chunck1,ans_chunck2])

    if showInfo: print(ans_chunck1.shape,ans_chunck2.shape,ref_chunck1.shape,ref_chunck2.shape)

    if showInfo: show_images([ans_chunck1])

    circles_chunck1 = Hough(ans_chunck1) 
    circles_chunck2 = Hough(ans_chunck2)   
    coordinates_chunk1 = show_Hough(ans_chunck1,circles_chunck1, showInfo)
    coordinates_chunk2 = show_Hough(ans_chunck2,circles_chunck2, showInfo)

    modelAnwsers = loadModelAnswer(modelAnswerPath)

    chunkAnswers1 = getAnswers(coordinates_chunk1,ans_chunck1,showInfo)
    chunkAnswers2 = getAnswers(coordinates_chunk2,ans_chunck2,showInfo)

    modelAnwsers = np.array(modelAnwsers)
    currAnswers = np.concatenate((chunkAnswers1, chunkAnswers2))

    if showInfo: print('Model answers: ',modelAnwsers)
    if showInfo: print('Current answers: ',currAnswers)
    print('Current answers: ')
    for i in range(currAnswers.shape[0]):
        print('Q'+str(i+1)+' = ' + str(currAnswers[i]))
    grade = (modelAnwsers == currAnswers)
    grade = grade.sum() / len(grade)
    if showInfo: print('Grade: ' + str(grade.sum()) + '/' + str(len(grade)))
    
    return flag, modelAnwsers, currAnswers, grade
