# import the necessary packages
from imutils import face_utils
import dlib
import cv2
from scipy.spatial import distance as dist

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

blink_thresh = 0.45
succ_frame = 4
count_frame = 0
n = 0

# Eye landmarks
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']


def calculate_EAR(eye):

    # calculate the vertical distances
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])

    # calculate the horizontal distance
    x1 = dist.euclidean(eye[0], eye[3])

    # calculate the EAR
    EAR = (y1+y2) / x1
    return EAR


cap = cv2.VideoCapture(0)

while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        lefteye = shape[L_start: L_end]
        righteye = shape[R_start:R_end]

        left_EAR = calculate_EAR(lefteye)
        right_EAR = calculate_EAR(righteye)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            # Avg of left and right eye EAR
            avg = (left_EAR+right_EAR)/2
            if avg < blink_thresh:
                count_frame += 1  # incrementing the frame count
            else:
                if count_frame >= succ_frame:
                    print("Blink Detected")
                    n = n + 1
                else:
                    count_frame = 0
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
