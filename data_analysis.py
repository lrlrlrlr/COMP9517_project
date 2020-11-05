# import
import cv2 as cv
import os
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
# load the images
from ultilities import *

# print the shape
arab_imgs = []
toba_imgs = []
img_shape = dict()


for filename in arab_filenames:
    img = cv.imread(path_arab+'\\'+filename,0)
    print(filename,' : ',img.shape)
    img_shape[filename] = img.shape

    img = cv.resize(img,(300,300))
    img = img.flatten()
    arab_imgs.append(img)
for filename in toba_filenames:
    img = cv.imread(path_toba+'\\'+filename,0)
    print(filename,' : ',img.shape)
    img_shape[filename] = img.shape

    img = cv.resize(img,(300,300))
    img = img.flatten()
    toba_imgs.append(img)

imgs=np.array(arab_imgs+toba_imgs)
img_labels =np.array(len(arab_imgs)*[True]+ len(toba_imgs)*[False]).reshape(-1,1) # True means arabidopsis, False means tobacco
