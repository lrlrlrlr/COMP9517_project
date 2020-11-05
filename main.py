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
from ultilities import imgs,img_labels
X_train,X_test,y_train,y_test = train_test_split(imgs,img_labels, test_size=0.25, shuffle=True)

# preprocessing
# get the features(in same dimension) imagearray + metadata



# train models
# compare different clf
classifiers = [KNeighborsClassifier(),
               DecisionTreeClassifier(),
               LinearDiscriminantAnalysis(),
               LogisticRegression(),
               GaussianNB(),
               SVC()]

classifier_accuracy_list = []
for i, classifier in enumerate(classifiers):
    # split the dataset into 5 folds; then test the classifier against each fold one by one
    accuracies = cross_val_score(classifier, imgs, img_labels.ravel(), cv=5)
    classifier_accuracy_list.append((accuracies.mean(), type(classifier).__name__))

# sort the classifiers
classifier_accuracy_list = sorted(classifier_accuracy_list, reverse=True)
for item in classifier_accuracy_list:
    print(item[1], ':', item[0])




