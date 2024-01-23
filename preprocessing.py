from PIL import Image, ImageFile, ImageOps
import matplotlib.pyplot as plt
import pathlib
import os
import cv2
import dlib
import pandas as pd
import numpy as np
import math
import pathlib


def preprocess_SVM(image):


    def get_lum(image, x, y, w, h, k):


        i1 = range(int(-w / 2), int(w / 2))
        j1 = range(0, h)

        lumar = np.zeros((len(i1), len(j1)))
        for i in i1:
            for j in j1:
                if y + k * h < 0 or y + k * h >= image.shape[0]:
                    lumar[i][j] = None
                else:
                    lum = np.min(np.clip(image[y + k * h, x + i], 0, 255))
                    lumar[i][j] = lum

        return np.min(lumar)


    def q(landmarks, index1, index2):

        x1 = landmarks[int(index1)][0]
        y1 = landmarks[int(index1)][1]
        x2 = landmarks[int(index2)][0]
        y2 = landmarks[int(index2)][1]

        x_diff = float(x1 - x2)


        if y1 < y2: y_diff = float(np.absolute(y1 - y2))
        if y1 >= y2:
            y_diff = 0.1

        return np.absolute(math.atan(x_diff / y_diff))


    def d(landmarks, index1, index2):
        # get distance between i1 and i2

        x1 = landmarks[int(index1)][0]
        y1 = landmarks[int(index1)][1]
        x2 = landmarks[int(index2)][0]
        y2 = landmarks[int(index2)][1]

        x_diff = (x1 - x2) ** 2
        y_diff = (y1 - y2) ** 2

        dist = math.sqrt(x_diff + y_diff)

        return dist

    ImageFile.LOAD_TRUNCATED_IMAGES = True


    img = cv2.imread(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
    img_f = cv2.resize(img_eq, (500, 600))

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    mn = 14
    faces = faceCascade.detectMultiScale(img_f, scaleFactor=1.1, minNeighbors=mn, minSize=(100, 100),
                                                 flags=cv2.CASCADE_SCALE_IMAGE)
    detector_flag = False
    flag2 = 0

    x, y, w, h = 0, 0, 0, 0
    while len(faces) != 1:
        prev_mn = mn
        if len(faces) == 0:
            faces = faceCascade.detectMultiScale(img_f, scaleFactor=1.1, minNeighbors=mn, minSize=(100, 100),
                                                         flags=cv2.CASCADE_SCALE_IMAGE)
            mn -= 1
            if mn == 0 and len(faces) == 0:
                faces = detector(img_f)
                detector_flag = True
                if len(faces) == 0:
                    x = 10
                    y = 10
                    w = img_f.shape[1] - 10
                    h = img_f.shape[0] - 10
                    detector_flag = False
                break
        else:
            faces = faceCascade.detectMultiScale(img_f, scaleFactor=1.1, minNeighbors=mn, minSize=(100, 100),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
            mn += 1

        if prev_mn < mn:
            flag2 = 1
            break

    if mn == 0 and detector_flag is True:
        face = faces[flag2]
        x, y = face.left(), face.top()
        w, h = face.right() - x, face.bottom() - y
    elif len(faces) != 0 and mn >= 0:
        face = faces[flag2]
        x, y, w, h = face

    cv2.rectangle(img_f, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow("Detected Faces", img_f)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    dlib_rect = dlib.rectangle(int(0.8 * x), int(0.8 * y), int(x + 1.05*w), int(y + 1.1*h))

    detected_landmarks = predictor(img_f, dlib_rect).parts()

    landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

            # print(landmarks)

    image_copy = img_f.copy()

    for idx, point in enumerate(landmarks):
        # print(point[0, 0], point[0, 1])
        pos = (point[0, 0], point[0, 1])

        cv2.circle(image_copy, pos, 5, color=(255, 0, 0), thickness=2)



    p27 = (landmarks[27][0, 0], landmarks[27][0, 1])
    x = p27[0]
    y = p27[1]

    diff = get_lum(img_f, x, y, 8, 2, -1)
    limit = diff - 55

    while diff > limit:
        y = int(y - 1)
        diff = get_lum(img_f, x, y, 6, 2, -1)

    cv2.circle(image_copy, (x, y), 5, color=(0, 0, 255), thickness=2)
    # cv2.imshow("Images_with_Landmarks", image_copy)
    # if cv2.waitKey(100) & 0XFF == 27:
    #     break
    # cv2.destroyAllWindows()

    lmark = landmarks.tolist()
    p68 = (x, y)
    lmark.append(p68)

    f = []

    fwidth = d(lmark, 0, 16)
    fheight = d(lmark, 8, 68)
    f.append(fheight / fwidth)

    jwidth = d(lmark, 4, 12)
    f.append(jwidth / fwidth)

    hchinmouth = d(lmark, 57, 8)
    f.append(hchinmouth / fwidth)
    ref = q(lmark, 27, 8)

    for k in range(0, 17):
        if k != 8:
            theta = q(lmark, k, 8)
            f.append(theta)

    for k in range(1, 8):
        dist = d(lmark, k, 16-k)
        f.append(dist/fwidth)

    return f









