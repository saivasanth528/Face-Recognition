import cv2
import sys

imagePath = 'images/vasanth_test.jpg'
cascPath = 'haarcascade_frontalface_default.xml'

# Creating the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detecting faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

'''
This function detects the actual face and is the key part of our code, so let’s go over the options:

The detectMultiScale function is a general function that detects objects. Since we are calling 
it on the face cascade, that’s what it detects.

The first option is the grayscale image.

The second is the scaleFactor. Since some faces may be closer to the camera, 
they would appear bigger than the faces in the back. The scale factor compensates for this.

The detection algorithm uses a moving window to detect objects. minNeighbors defines
how many objects are detected near the current one before it declares the face found.
minSize, meanwhile, gives the size of each window.

'''

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
