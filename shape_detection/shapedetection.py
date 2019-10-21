import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("E:\Computer Vision\Images\sign.png",1)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower = np.array([0,0,0])
upper = np.array([15,250,255])
lower1 = np.array([165,0,0])
upper1 = np.array([179,250,255])

mask1 = cv2.inRange(hsv,lower,upper)
mask2 = cv2.inRange(hsv,lower1,upper1)
mask = mask1+mask2 #OR operation 
out = cv2.bitwise_and(img,img,mask = mask)

gray = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
med = cv2.medianBlur(gray,5) #removing noise
hough = cv2.HoughCircles(med,cv2.HOUGH_GRADIENT,8,img.shape[0]/64,param1=200,param2 = 150,minRadius=2,maxRadius=300)
if hough is not None:
    for i in hough[0, :]:
        cv2.circle(out, (i[0], i[1]), i[2], (255, 255, 0), 2)
cv2.imshow("final",out)
cv2.waitKey()
cv2.destroyAllWindows()