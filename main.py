# Assignment 3.m
# Name: Zaid M. Samamah
# Date: 22/12/2022
# Semester: Fall 2022
# Computer Vision Course
import math
import numpy as np
from sympy import *
import cv2 as cv
import sys

# 1
picture = "images/picture.png"
cam = cv.VideoCapture(0)
while True:
    ret_val, img = cam.read()
    if cv.waitKey(1) == 27:
        cv.imwrite(picture, img)
        break
    elif img is None:
        sys.exit('Couldn`t read the image')
        break
    cv.imshow('My Webcam', img)

del cam

# 2
img = cv.imread(picture)
cropped_image = img[50:350, 150:500]
cv.imwrite("images/Cropped Image.png", cropped_image)
cropped_image = cv.imread("images/Cropped Image.png")
proj1 = cv.imread("images/1.jpg")
proj2 = cv.imread("images/2.jpg")
proj3 = cv.imread("images/3.jpg")
cv.imshow("cropped", cropped_image)

cv.waitKey(0)

# 3
A1 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, -788],
               [0, 0, 0, 0, 0, 1, 0, 0, -190],
               [350, 0, 1, 0, 0, 0, -381500, 0, -1090],
               [0, 0, 0, 350, 0, 1, -74550, 0, -213],
               [0, 300, 1, 0, 0, 0, 0, -229500, -765],
               [0, 0, 0, 0, 300, 1, 0, -263100, -877],
               [350, 300, 1, 0, 0, 0, -369250, -316500, -1055],
               [0, 0, 0, 350, 300, 1, -269500, -231000, -770]]
              )
A1 = Matrix(A1)
H1 = Matrix.nullspace(A1)
H1 = np.reshape(H1, (3, 3)).astype(float)
print(H1)

# 5
A2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, -601],
               [0, 0, 0, 0, 0, 1, 0, 0, -373],
               [350, 0, 1, 0, 0, 0, -333200, 0, -952],
               [0, 0, 0, 350, 0, 1, -133350, 0, -381],
               [0, 300, 1, 0, 0, 0, 0, -177900, -593],
               [0, 0, 0, 0, 300, 1, 0, -249900, -833],
               [350, 300, 1, 0, 0, 0, -328300, -281400, -938],
               [0, 0, 0, 350, 300, 1, -287700, -246600, -822]])
A2 = Matrix(A2)
H2 = Matrix.nullspace(A2)
H2 = np.reshape(H2, (3, 3)).astype(float)
print(H2)

# 6
A3 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, -536],
               [0, 0, 0, 0, 0, 1, 0, 0, -157],
               [350, 0, 1, 0, 0, 0, -280000, 0, -800],
               [0, 0, 0, 350, 0, 1, -52500, 0, -150],
               [0, 300, 1, 0, 0, 0, 0, -160200, -534],
               [0, 0, 0, 0, 300, 1, 0, -204300, -681],
               [350, 300, 1, 0, 0, 0, -274400, -235200, -784],
               [0, 0, 0, 350, 300, 1, -272300, -233400, -778]]
              )
A3 = Matrix(A3)
H3 = Matrix.nullspace(A3)
H3 = np.reshape(H3, (3, 3)).astype(float)
print(H3)

src_img = cv.imread("images/Cropped Image.png")
src_pts = np.array([[0, 0], [350, 0], [0, 300], [350, 300]])
tgt_pts = np.array([[788, 190], [1090, 213], [765, 877], [1055, 770]])
M, mask = cv.findHomography(src_pts, tgt_pts)
h, w = tgt_pts.shape[:2]

for row in range(300):
    for col in range(350):
        #1
        pt = np.dot(H1, np.array([col, row, 1]))
        pt = pt/pt[2]
        x = math.floor(pt[0])
        y = math.floor(pt[1])
        pixel = cropped_image[row, col, :]
        try:
            proj1[y, x, :] = pixel
            proj1[y+1, x+1, :] = pixel
        except IndexError:
            continue
        #2
        pt = np.dot(np.reshape(H2, (3, 3)).astype(float), np.array([col, row, 1]))
        pt = pt / pt[2]
        x = math.floor(pt[0])
        y = math.floor(pt[1])
        pixel = cropped_image[row, col, :]
        try:
            proj2[y, x, :] = pixel
            proj2[y + 1, x + 1, :] = pixel
        except IndexError:
            continue
        #3
        pt = np.dot(np.reshape(H3, (3, 3)).astype(float), np.array([col, row, 1]))
        pt = pt / pt[2]
        x = math.floor(pt[0])
        y = math.floor(pt[1])
        pixel = cropped_image[row, col, :]
        try:
            proj3[y, x, :] = pixel
            proj3[y + 1, x + 1, :] = pixel
        except IndexError:
            continue

cv.imwrite("images/proj1.png", proj1)
cv.imwrite("images/proj2.png", proj2)
cv.imwrite("images/proj3.png", proj3)