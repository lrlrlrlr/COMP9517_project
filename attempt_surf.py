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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import matplotlib.pyplot as plt
# load the images
from ultilities import *

imgs,img_labels = surf_features()
X_train,X_test,y_train,y_test = train_test_split(imgs,img_labels, test_size=0.25, shuffle=True)

# preprocessing
# get the features(in same dimension) imagearray + metadata



# train models
# compare different clf
classifiers = [
    KNeighborsClassifier(),
    SVC(),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
    ]

classifier_accuracy_list = []
for i, classifier in enumerate(classifiers):
    # split the dataset into 5 folds; then test the classifier against each fold one by one
    accuracies = cross_val_score(classifier, imgs, img_labels.ravel(), cv=5)
    classifier_accuracy_list.append((accuracies.mean(), type(classifier).__name__))

# sort the classifiers
classifier_accuracy_list = sorted(classifier_accuracy_list, reverse=True)
for item in classifier_accuracy_list:
    print(item[1], ':', item[0])






