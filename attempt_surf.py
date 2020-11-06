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
import warnings

import matplotlib.pyplot as plt
# load the images
from ultilities import *


def compare_diff_models(X, y):
    # # train models
    # # compare different clf
    classifiers = [
        KNeighborsClassifier(),
        SVC(),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
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
        accuracies = cross_val_score(classifier, X, y.ravel(), cv=5)
        classifier_accuracy_list.append((accuracies.mean(), type(classifier).__name__))

    # sort the classifiers
    # classifier_accuracy_list = sorted(classifier_accuracy_list, reverse=True)
    classifier_accuracy_list = sorted(classifier_accuracy_list, reverse=True)
    for item in classifier_accuracy_list:
        print(item[1], ':', item[0])


if __name__ == '__main__':
    # for n in [1000,500,1500]:
    #     print("hessianThreshold: ", n)
    imgs, img_labels = surf_features(segmented=True)
    X_train, X_test, y_train, y_test = train_test_split(np.array(imgs), np.array(img_labels), test_size=0.25,
                                                        shuffle=True)

    clf = SVC()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("accuracy:\t", accuracy_score(y_test, predictions))
    print("recall:\t\t", recall_score(y_test, predictions, average='macro'))

    from sklearn.metrics import roc_auc_score
    y_scores = clf.decision_function(X_test)
    print("AUC:", roc_auc_score(y_test, y_scores))

    from sklearn.metrics import f1_score

    print("f1:", f1_score(y_test, predictions, zero_division=1))
