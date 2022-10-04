import math
from imutils import face_utils
import dlib
import os
import cv2


bCount = 2


def blinkCount(num):
    res = "Blink Counter = " + str(num)
    return res


BLINK_RATIO_THRESHOLD = 5.7

p = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


def midpoint(point1, point2):
    return int((point1[0] + point2[0])/2), int((point1[1] + point2[1])/2)


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_blink_ratio(eye_points):

    # loading all the required points
    corner_left = (eye_points[0][0], eye_points[0][1])
    corner_right = (eye_points[3][0], eye_points[3][1])

    center_top = midpoint(eye_points[1], eye_points[2])
    center_bottom = midpoint(eye_points[5], eye_points[4])

    # calculating distance
    horizontal_length = euclidean_distance(corner_left, corner_right)
    vertical_length = euclidean_distance(center_top, center_bottom)

    ratio = horizontal_length / vertical_length

    return ratio


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Video.mp4')

while True:

    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # DETECTIONG FACES
    rects = detector(gray, 0)

    # loop over the face detections
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        cor = [slice(36, 42), slice(42, 48), slice(36, 48)]

        # GET BLINK RATIO
        left_eye_ratio = get_blink_ratio(shape[cor[1]])
        right_eye_ratio = get_blink_ratio(shape[cor[0]])
        blink_ratio = (left_eye_ratio + right_eye_ratio) / 2
        # OMPARE BLINK RATIO
        if blink_ratio > BLINK_RATIO_THRESHOLD:
            cv2.putText(image, blinkCount(bCount), (30, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 100, 255), 1)
            bCount += 1

        # show the output image with facial landmarks
        '''
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        '''

    cv2.imshow("Eye Blink Counter", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
