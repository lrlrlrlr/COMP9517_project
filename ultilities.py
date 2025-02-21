# COMP9517 PROJECT - INDIVIDUAL PART
# ultilities.py : access the program dataset, preprocessing, feature extraction, compare models
# AUTHOR: Rui Li
# Email: z5202952@ad.unsw.edu.au

import os
import warnings

import cv2 as cv
import numpy as np
# import
from skimage.feature import hog
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings(action='ignore', category=DataConversionWarning)


# load the images
def load_files(segmented=False):
    path_arab = "D:\OneDrive - UNSW\COMP9517\project\Plant\Ara2013-Canon"
    path_toba = "D:\OneDrive - UNSW\COMP9517\project\Plant\Tobacco"
    path_toba_segmented = "segmented"
    toba_filenames_segmented = [pic for pic in os.listdir(path_toba_segmented)]
    toba_filenames = [pic for pic in os.listdir(path_toba) if '_rgb.png' in pic]
    arab_filenames = [pic for pic in os.listdir(path_arab) if
                      '_rgb.png' in pic]
    if segmented:
        return path_toba_segmented, toba_filenames_segmented, path_arab, arab_filenames
    return path_toba, toba_filenames, path_arab, arab_filenames


def resize_img(imgsize=(50 * 50), segmented=False):
    arab_imgs = []
    toba_imgs = []
    path_toba, toba_filenames, path_arab, arab_filenames = load_files(segmented)
    for filename in arab_filenames:
        img = cv.imread(path_arab + '\\' + filename, 0)
        img = cv.resize(img, imgsize)
        img = img.flatten()
        arab_imgs.append(img)

    for filename in toba_filenames:
        img = cv.imread(path_toba + '\\' + filename, 0)

        img = cv.resize(img, imgsize)
        img = img.flatten()
        toba_imgs.append(img)

    imgs = np.array(arab_imgs + toba_imgs)
    img_labels = np.array(len(arab_imgs) * [True] + len(toba_imgs) * [False]).reshape(-1,
                                                                                      1)  # True means arabidopsis, False means tobacco
    return imgs, img_labels  # return image as feature


def hog_img(segmented=True):
    path_toba, toba_filenames, path_arab, arab_filenames = load_files(segmented)
    # load the img
    features = []
    labels_is_tobbaco = []
    for filename in toba_filenames:
        img = cv.imread(path_toba + '\\' + filename)
        fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, multichannel=True)
        features.append(hog_image.flatten())
        labels_is_tobbaco.append(True)

    for filename in arab_filenames:
        img = cv.imread(path_arab + '\\' + filename)
        fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, multichannel=True)
        features.append(hog_image.flatten())
        labels_is_tobbaco.append(False)
    return np.array(features), np.array(labels_is_tobbaco)


def surf_features(hessianThreshold=1000, segmented=True):
    path_toba, toba_filenames, path_arab, arab_filenames = load_files(segmented)

    features = []
    labels_is_tobbaco = []

    surf = cv.xfeatures2d.SURF_create(hessianThreshold)

    for filename in toba_filenames:
        img = cv.imread(path_toba + '\\' + filename, 0)
        kp, des = surf.detectAndCompute(img, None)
        try:
            if len(des[:5].flatten()) == 320:
                features.append(des[:5].flatten())
                labels_is_tobbaco.append(True)
        except TypeError:
            pass

    for filename in arab_filenames:
        img = cv.imread(path_arab + '\\' + filename, 0)
        kp, des = surf.detectAndCompute(img, None)
        features.append(des[:5].flatten())
        labels_is_tobbaco.append(False)

    return features, labels_is_tobbaco


def split_data(X, y, split_percentage=0.75):
    split_point = int(len(X) * split_percentage)
    X_train = X[:split_point]
    y_train = y[:split_point]
    X_test = X[split_point:]
    y_test = y[split_point:]
    return X_train, y_train, X_test, y_test


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
    pass
