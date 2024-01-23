import cv2
import dlib
import pandas as pd
import numpy as np
import math
import pathlib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, classification_report, roc_curve, auc, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import pickle

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




#extracting training data features

image_dir = "FaceShapeDatasetProcessingOutput/training_set"
cascade_path = "haarcascade_frontalface_default.xml"
predictor_path = "shape_predictor_68_face_landmarks.dat"

faceCascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

sub_dir = [q for q in pathlib.Path(image_dir).iterdir() if q.is_dir()]
dirs = ["Heart", "Oblong", "Oval", "Round", "Square"]

features = []

for i in range(len(sub_dir)):
    images_dir = [p for p in pathlib.Path(sub_dir[i]).iterdir() if p.is_file()]
    for j in range(len(images_dir)):

        image = cv2.imread(str(images_dir[j]))
        image = cv2.resize(image, (500, 600))

        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        string = pathlib.Path(images_dir[j]).name
        print(string)

        mn = 14
        faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=mn, minSize=(100, 100),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        detector_flag = False
        flag2 = 0
        # print(string)
        x, y, w, h = 0, 0, 0, 0
        while len(faces) != 1:
            prev_mn = mn
            if len(faces) == 0:
                faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=mn, minSize=(100, 100),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
                mn -= 1
                if mn == 0 and len(faces) == 0:
                    faces = detector(image)
                    detector_flag = True
                    if len(faces) == 0:
                        x = 10
                        y = 10
                        w = image.shape[1] - 10
                        h = image.shape[0] - 10
                        detector_flag = False
                    break
            else:
                faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=mn, minSize=(100, 100),
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

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow("Detected Faces", image)
        # if cv2.waitKey(500) & 0XFF == 27:
        #     break
        # cv2.destroyAllWindows()

        # print(faces)

        dlib_rect = dlib.rectangle(int(0.8 * x), int(0.8 * y), int(x + 1.05*w), int(y + 1.1*h))

        detected_landmarks = predictor(image, dlib_rect).parts()

        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

        # print(landmarks)

        image_copy = image.copy()

        for idx, point in enumerate(landmarks):
            # print(point[0, 0], point[0, 1])
            pos = (point[0, 0], point[0, 1])

            cv2.circle(image_copy, pos, 5, color=(255, 0, 0), thickness=2)

        # cv2.imshow("Images_with_Landmarks", image_copy)
        # if cv2.waitKey(25) & 0XFF == 27:
        #     break
        # cv2.destroyAllWindows()

        p27 = (landmarks[27][0, 0], landmarks[27][0, 1])
        x = p27[0]
        y = p27[1]

        diff = get_lum(image, x, y, 8, 2, -1)
        limit = diff - 55

        while diff > limit:
            y = int(y - 1)
            diff = get_lum(image, x, y, 6, 2, -1)

        cv2.circle(image_copy, (x, y), 5, color=(0, 0, 255), thickness=2)
        # cv2.imshow("Images_with_Landmarks", image_copy)
        # if cv2.waitKey(100) & 0XFF == 27:
        #     break
        # cv2.destroyAllWindows()

        lmark = landmarks.tolist()
        p68 = (x, y)
        lmark.append(p68)

        f = []
        f.append(dirs[i])

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

        features.append(f)


image_dir = "FaceShapeDatasetProcessingOutput/testing_set"
cascade_path = "haarcascade_frontalface_default.xml"
predictor_path = "shape_predictor_68_face_landmarks.dat"
test_features = []

for i in range(len(sub_dir)):
    images_dir = [p for p in pathlib.Path(sub_dir[i]).iterdir() if p.is_file()]
    for j in range(len(images_dir)):

        image = cv2.imread(str(images_dir[j]))
        image = cv2.resize(image, (500, 600))

        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        string = pathlib.Path(images_dir[j]).name

        print(string)

        mn = 14
        faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=mn, minSize=(100, 100),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        detector_flag = False
        flag2 = 0
        # print(string)
        x, y, w, h = 0, 0, 0, 0
        while len(faces) != 1:
            prev_mn = mn
            if len(faces) == 0:
                faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=mn, minSize=(100, 100),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
                mn -= 1
                if mn == 0 and len(faces) == 0:
                    faces = detector(image)
                    detector_flag = True
                    if len(faces) == 0:
                        x = 10
                        y = 10
                        w = image.shape[1] - 10
                        h = image.shape[0] - 10
                        detector_flag = False
                    break
            else:
                faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=mn, minSize=(100, 100),
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

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow("Detected Faces", image)
        # if cv2.waitKey(500) & 0XFF == 27:
        #     break
        # cv2.destroyAllWindows()

        # print(faces)

        dlib_rect = dlib.rectangle(int(0.8 * x), int(0.8 * y), int(x + 1.05*w), int(y + 1.1*h))

        detected_landmarks = predictor(image, dlib_rect).parts()

        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

        # print(landmarks)

        image_copy = image.copy()

        for idx, point in enumerate(landmarks):
            # print(point[0, 0], point[0, 1])
            pos = (point[0, 0], point[0, 1])

            cv2.circle(image_copy, pos, 5, color=(255, 0, 0), thickness=2)

        # cv2.imshow("Images_with_Landmarks", image_copy)
        # if cv2.waitKey(25) & 0XFF == 27:
        #     break
        # cv2.destroyAllWindows()

        p27 = (landmarks[27][0, 0], landmarks[27][0, 1])
        x = p27[0]
        y = p27[1]

        diff = get_lum(image, x, y, 8, 2, -1)
        limit = diff - 55

        while diff > limit:
            y = int(y - 1)
            diff = get_lum(image, x, y, 6, 2, -1)

        cv2.circle(image_copy, (x, y), 5, color=(0, 0, 255), thickness=2)
        # cv2.imshow("Images_with_Landmarks", image_copy)
        # if cv2.waitKey(100) & 0XFF == 27:
        #     break
        # cv2.destroyAllWindows()

        lmark = landmarks.tolist()
        p68 = (x, y)
        lmark.append(p68)

        f = []
        f.append(dirs[i])

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

        test_features.append(f)

x_train = [feature[1:] for feature in features]
x_test = [feature[1:] for feature in test_features]
y_train = [feature[0] for feature in features]
y_test = [feature[0] for feature in test_features]
print(x_test)

# print(x_train)
# print("---------------------------------------------------------------------------------------------------------------")
# print(x_test)
# print("---------------------------------------------------------------------------------------------------------------")
# print(y_train)
# print("---------------------------------------------------------------------------------------------------------------")
# print(y_test)

param_grid = {
    'C': np.logspace(-5, 5, 15),
    'gamma': np.logspace(-5, 5, 15)
}
#
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
print(grid_search)
model = grid_search.fit(x_train, y_train)
print("done")
#
best_params = grid_search.best_params_
print(best_params)
best_model = grid_search.best_estimator_
print(best_model)
#
print("Best Hyperparameters:", best_params)
y_pred = best_model.predict(x_test)
y_predicted_proba = best_model.predict_proba(x_test)
print(y_pred, y_predicted_proba)
#
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
classification_rep = classification_report(y_test, y_pred)
#

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dirs, yticklabels=dirs)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

print('Accuracy: ', accuracy)
print("Recall: ", recall)
print("Precision: ", precision)
print("F1: ", f1)
print("Classification_report: \n", classification_rep)


y_test_five = label_binarize(y_test, classes=np.unique(y_test))
y_pred_five = label_binarize(y_pred, classes=np.unique(y_test))

classes = 5
fpr = []
tpr = []
roc_auc = []
for i in range(classes):
    fpr_i, tpr_i, _ = roc_curve(y_test_five[:, i], y_pred_five[:, i])
    roc_auc_i = auc(fpr_i, tpr_i)
    fpr.append(fpr_i)
    tpr.append(tpr_i)
    roc_auc.append(roc_auc_i)

for i in range(classes):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc[i])


plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()



with open("SVM_model.pickle", "wb") as model_file:
    pickle.dump(model, model_file)








