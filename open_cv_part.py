# OpenCV program to detect face in real time
# import libraries of python OpenCV
# where its functionality resides
import cv2
from Face_Recognition import *
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
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# capture frames from a camera
cap = cv2.VideoCapture(0)
# cap.set(3, 80)  # WIDTH
# cap.set(4, 80)  # HEIGHT

# loop runs if capturing has been initialized.

def paint_detected_face_on_image(frame, location, name=None):
    """
    Paint a rectangle around the face and write the name
    """
    # unpack the coordinates from the location tuple
    top, right, bottom, left = location

    if name is None:
        name = 'Unknown'
        color = (0, 0, 255)  # red for unrecognized face
    else:
        color = (0, 128, 0)  # dark green for recognized face

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)


def start_surveilance():
    count = 0
    while 1:

        # reads frames from a camera
        ret, img = cap.read()

        # convert to gray scale of each frames
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("frame%d.jpg" % count, img)
        # Detects faces of different sizes in the input image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(faces)
        if len(faces):
            face_frame = cv2.resize(img, (96, 96))
            min_distance, identity = who_is_it(face_frame, database, FRmodel)

            for (x, y, w, h) in faces:
                # To draw a rectangle in a face
                if min_distance is not None:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 128, 0), 2) # green
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]

                # Detects eyes of different sizes in the input image
                # eyes = eye_cascade.detectMultiScale(roi_gray)
                #
                # # To draw a rectangle in eyes
                # for (ex, ey, ew, eh) in eyes:
                #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 127, 255), 2)

            # Display an image in a window

            # paint_detected_face_on_image(img, *faces, identity)
        cv2.imshow('img', img)
        count += 1
        faces = ()
        # Wait for Esc key to stop
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # Close the window
    cap.release()

    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


if __name__ == '__main__':

    print("**************SMART SURVEILANCE MODEL***************")
    print("1.ADD AUTHORIZED PERSON")
    print("2.DELETE AUTHORIZED PERSON")
    print("3.START SURVEILANCE")

    choosen_option = int(input())

    if choosen_option == 1:
        pass
    elif choosen_option == 2:
        pass
    elif choosen_option == 3:
        start_surveilance()
