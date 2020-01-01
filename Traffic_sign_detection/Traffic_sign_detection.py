import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

avoid_repeat_sign = 0;
detected_sign_image = []; #image to be recognised stored here

cap = cv2.VideoCapture("F:/IEEE_Project_2019/data_collection/demo.gif")
while True:
   
	#reading video frame
	ret,frame = cap.read()
	if(not(ret)):
		break
	img_sign = cv2.resize(frame,(400,400))
    
	
	#color thresholding
	img_hsv = cv2.cvtColor(img_sign , cv2.COLOR_BGR2HSV)
	min_hsv_blue = np.array([90,100,50])
	max_hsv_blue = np.array([120,255,255])
	threshold1 = cv2.inRange(img_hsv , min_hsv_blue , max_hsv_blue)
	
	
	'''
	#code for detecting red rim sign(out of scope for now)
	
	min_hsv_red1 = np.array([170,0,0])
	max_hsv_red1 = np.array([190,255,255])
	threshold2 = cv2.inRange(img_hsv , min_hsv_red1 , max_hsv_red1)
	min_hsv_red2 = np.array([0,100,100])
	max_hsv_red2 = np.array([10,255,255])
	threshold3 = cv2.inRange(img_hsv , min_hsv_red2 , max_hsv_red2)
	threshold = cv2.bitwise_or(threshold1, threshold2 , threshold3)
	'''
	
	
	
	# Morphological operation
	thres = cv2.dilate(threshold1,(5,5))
	thres = cv2.dilate(thres,(5,5))
	thres = cv2.erode(thres,(5,5))
	thres = cv2.erode(thres,(5,5))

	
	#Traffic sign detection
	_,cnts,_ = cv2.findContours(thres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	E=0
	for i in cnts:
		area = cv2.contourArea(i)
		if(area>1000 and avoid_repeat_sign == 0):
			M = cv2.moments(i)
			_,(ma,mb),_ = cv2.fitEllipse(i)
			E = M['m00']/(math.pi*(ma/2)*(mb/2))
			if(E>0.95):
				avoid_repeat_sign = 20    # no sign would be detected for next 20 frame
				x,y,w,h = cv2.boundingRect(i)
				detected_sign_image = img_sign[y:y+h,x:x+w]
				cv2.rectangle(img_sign, (x,y), (x+w,y+h), (0,255,255), 3)
	avoid_repeat_sign = max(--avoid_repeat_sign , 0)			
	cv2.imshow('sign',img_sign)
	cv2.waitKey(50)
    
cv2.destroyAllWindows()
cap.release()

