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
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.exceptions import DataConversionWarning
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# load the images
path_arab = "D:\OneDrive - UNSW\COMP9517\project\Plant\Ara2013-Canon"
path_toba = "D:\OneDrive - UNSW\COMP9517\project\Plant\Tobacco"
path_toba_segmented = "segmented"
toba_filenames_segmented = [pic for pic in os.listdir(path_toba_segmented)]
toba_filenames = [pic for pic in os.listdir("D:\OneDrive - UNSW\COMP9517\project\Plant\Tobacco") if '_rgb.png' in pic]
arab_filenames = [pic for pic in os.listdir("D:\OneDrive - UNSW\COMP9517\project\Plant\Ara2013-Canon") if
                  '_rgb.png' in pic]


def resize_img(imgsize=(50 * 50)):
    arab_imgs = []
    toba_imgs = []

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
    return imgs, img_labels # return image as feature

def hog_img(segmented=True):
    if segmented:
        filenames = toba_filenames_segmented
        path = path_toba_segmented
    else:
        filenames = toba_filenames
        path = path_toba
    # load the img
    features = []
    labels_is_tobbaco = []
    for filename in filenames:
        img = cv.imread(path + '\\' + filename)
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
    return np.array(features),np.array(labels_is_tobbaco)

def surf_features(hessianThreshold=1000, segmented=True):
    if segmented:
        filenames = toba_filenames_segmented
        path = path_toba_segmented
    else:
        filenames = toba_filenames
        path = path_toba

    features = []
    labels_is_tobbaco = []

    surf = cv.xfeatures2d.SURF_create(hessianThreshold)

    for filename in filenames:
        img = cv.imread(path + '\\' + filename, 0)
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

    return features,labels_is_tobbaco

def split_data(X,y,split_percentage=0.75):
    split_point = int(len(X) * split_percentage)
    X_train = X[:split_point]
    y_train = y[:split_point]
    X_test = X[split_point:]
    y_test = y[split_point:]
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    # X, y = hog_img()
    img = cv.imread(path_toba + '\\' + toba_filenames[0], 0)
    img_resized = cv.resize(img, (300 , 300))

    X,y = surf_features()