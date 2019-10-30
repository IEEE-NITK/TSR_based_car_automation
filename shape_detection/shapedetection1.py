#Implementing hough transform on a video(gif)

import cv2
import numpy as np

cap = cv2.VideoCapture("E:\Computer Vision\Images\demo.gif")

while True:
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray,(3,3),0,0)
        hough = cv2.HoughCircles(gauss,cv2.HOUGH_GRADIENT,1,80,param1=200,param2 = 28,minRadius=4,maxRadius=25)
        if hough is not None:
            hough = np.uint16(np.around(hough))
            for i in hough[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], (255, 255, 0), 2)
        cv2.imshow("out",frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    else:
        break
        
cap.release()
cv2.destroyAllWindows()