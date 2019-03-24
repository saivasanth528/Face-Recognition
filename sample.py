# import cv2
# img1 = cv2.imread("images/vasanth_test.jpg", 1)
# img1.shape
# img1 = img1[0:500, 0:500]
# cv2.imshow("cropped", img1)
# cv2.waitKey(0)
#
# import numpy as np
# import cv2
# faceCass = cv2.CascadeClassifier('face.xml')
# cam = cv2.VideoCapture(0)
# while True:
#     flag,img=cam.read()
#     grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     face=faceCass.detectMultiScale(grayImg,1.3,5)
#     for (x,y,w,h) in face :
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     cv2.imshow('face',img)
#     if cv2.waitKey(1)==27 :
#         break
# cv2.destroyAllWindows()
# cam.release()


# OpenCV program to detect face in real time
# import libraries of python OpenCV
# where its functionality resides
import cv2

# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# https://github.com/Itseez/opencv/blob/master
# /data/haarcascades/haarcascade_eye.xml
# Trained XML file for detecting eyes
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# capture frames from a camera
cap = cv2.VideoCapture(0)
count = 0
# loop runs if capturing has been initialized.
while 1:

	# reads frames from a camera
	ret, img = cap.read()

	# convert to gray scale of each frames
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detects faces of different sizes in the input image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		# To draw a rectangle in a face
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

	# Display an image in a window
	cv2.imwrite("frame%d.jpg" % count, img)
	cv2.imshow('img', img)
	count += 1
	# Wait for Esc key to stop
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
