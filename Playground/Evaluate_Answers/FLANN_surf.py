from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt
parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
parser.add_argument('--input1', help='Path to input image 1.', default='box.png')
parser.add_argument('--input2', help='Path to input image 2.', default='box_in_scene.png')
args = parser.parse_args()
img1 = cv.imread(cv.samples.findFile(args.input1), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(cv.samples.findFile(args.input2), cv.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)
#-- Step 1: Detect the keypoints using ORB Detector, compute the descriptors
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp1 = orb.detect(img1,None)
kp2 = orb.detect(img2,None)
# compute the descriptors with ORB
kp1, descriptors1 = orb.compute(img1, kp1)
kp2, descriptors2 = orb.compute(img2, kp2)

descriptors1=np.float32(descriptors1)
descriptors2=np.float32(descriptors2)

# draw only keypoints location,not size and orientation
new_img1 = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
plt.imshow(new_img1), plt.show()
new_img2 = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
plt.imshow(new_img2), plt.show()

#-- Step 2: Matching descriptor vectors with a FLANN based matcher
# Since SURF is a floating-point descriptor NORM_L2 is used
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
#-- Filter matches using the Lowe's ratio test
ratio_thresh = 0.7
good_matches = []
for m,n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
#-- Draw matches

#If enough matches are found, we extract the locations of matched keypoints in both the images. 
# They are passed to find the perspective transformation. Once we get this 3x3 transformation matrix
# , we use it to transform the corners of queryImage to corresponding points in trainImage

if len(good_matches)>10:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good_matches), 10) )
    matchesMask = None

    ## (8) drawMatches
matched = cv.drawMatches(img1,kp1,img2,kp2,good_matches,None)#,**draw_params)

## (9) Crop the matched region from scene
h,w = img1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv.perspectiveTransform(pts,M)
perspectiveM = cv.getPerspectiveTransform(np.float32(dst),pts)
found = cv.warpPerspective(img2,perspectiveM,(w,h))

cv.namedWindow( "matched", cv.WINDOW_NORMAL)
cv.namedWindow( "found", cv.WINDOW_NORMAL )

## (10) save and display
cv.imwrite("matched.png", matched)
cv.imwrite("found.png", found)
cv.imshow("matched", matched)
cv.imshow("found", found)
cv.waitKey();cv.destroyAllWindows()

