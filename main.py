# import the necessary packages
from distutils.command.clean import clean
from random import Random
import random
from imutils import face_utils
import dlib
import os
import cv2


ix = 0
tim = 0

p = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

FACIAL_LANDMARKS_IDXS = [
    # ("jaw", (0, 17))
    # ("right_eyebrow", (17, 22)),
    # ("left_eyebrow", (22, 27)),
    # ("nose", (27, 35)),
    # ("right_eye", (36, 42)),
    # ("left_eye", (42, 48)),
    # ("mouth", (48, 68)),
]

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Video.mp4')

while True:

    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)
    cv2.putText(image, 'Welcome to my world!!', (30, 30),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 100, 255), 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # if (ix < 7):
        #     if (random.randint(0, 9) % 2 != 0):
        #         ix = 4
        #     else:
        #         ix = 0
        # else:
        #     ix = 0
        ix = random.randint(4, 8)
        cor = [slice(0, 17, 1), slice(17, 22, 1), slice(22, 27, 1), slice(
            27, 35, 1), slice(36, 42, 1), slice(36, 48, 1),  slice(0, 0), slice(42, 48, 1), slice(0, 0), slice(48, 68, 1)]
        shape = shape[cor[ix]]
        for (x, y) in shape:
            cv2.circle(image, (x, y), 6, (255, 255, 255), -1)
            # Avg of left and right eye EAR
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Gautam Yadav", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
