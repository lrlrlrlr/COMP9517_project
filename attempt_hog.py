# COMP9517 PROJECT - INDIVIDUAL PART
# AUTHOR: Rui Li
# Email: z5202952@ad.unsw.edu.au

# import

# load the images
from ultilities import *

imgs, img_labels = hog_img(True)
X_train, X_test, y_train, y_test = train_test_split(imgs, img_labels, test_size=0.25, shuffle=True)

# preprocessing
# get the features(in same dimension) imagearray + metadata



# train models
# compare different clf
classifiers = [
    KNeighborsClassifier(3),
    SVC(gamma=2, C=1)]

classifier_accuracy_list = []
for i, classifier in enumerate(classifiers):
    # split the dataset into 5 folds; then test the classifier against each fold one by one
    accuracies = cross_val_score(classifier, imgs, img_labels.ravel(), cv=5)
    classifier_accuracy_list.append((accuracies.mean(), type(classifier).__name__))

# sort the classifiers
classifier_accuracy_list = sorted(classifier_accuracy_list, reverse=True)
for item in classifier_accuracy_list:
    print(item[1], ':', item[0])






