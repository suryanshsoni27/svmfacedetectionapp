from sklearn.decomposition import PCA
import pickle
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# loading model
haar = cv2.CascadeClassifier(
    'finalApp/model/haarcascade_frontalface_default.xml')

# pickle files
mean = pickle.load(open('finalApp/model/mean_preprocess.pickle', 'rb'))
model_svm = pickle.load(open('finalApp/model/model_svm.pickle', 'rb'))
model_pca = pickle.load(open('finalApp/model/pca_50.pickle', 'rb'))

# settings
gender_pre = ['male', 'female']
font = cv2.FONT_HERSHEY_SIMPLEX


def pipeline_model(path, filename, color='rgb'):
    # step 1 red image in cv2
    img = cv2.imread(path)
    # conv into gray scale
    if color == 'bgr':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # step4 - crop face
    faces = haar.detectMultiScale(gray, 1.5, 3)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        roi = gray[y:y+h, x:x+h]
        # normalization
        roi = roi/255.0

        # step 5 resize
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_CUBIC)

        roi_reshape = roi_resize.reshape(1, 10000)
        # step-7 subtract with meean

        roi_mean = roi_reshape - mean
        # step-8 get eigen image
        eigen_image = model_pca.transform(roi_mean)

        #step - 9
        results = model_svm.predict(eigen_image)[0]

        # step-10
        predict = results
        score = .82
        print(results)
        # step-10
        text = "%s : %0.2f" % (gender_pre[predict], score)
        cv2.putText(img, text, (x, y), font, 1, (255, 255, 0), 2)

    cv2.imwrite('finalApp/static/predict/{}'.format(filename), img)
