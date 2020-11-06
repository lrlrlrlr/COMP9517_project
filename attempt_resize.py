# COMP9517 PROJECT - INDIVIDUAL PART
# attempt_resize.py train model with imgs' raw pixel
# AUTHOR: Rui Li
# Email: z5202952@ad.unsw.edu.au


# import

# load the images
from ultilities import *

for seg in [True, False]:
    print("Segmented:", seg)
    for size in [50, 100, 200, 300]:
        # preprocessing
        print('Resize the image to: {}*{}'.format(size, size))
        imgs, img_labels = resize_img((size, size), seg)
        X_train, X_test, y_train, y_test = train_test_split(imgs, img_labels, test_size=0.25, shuffle=True)

        # compare_diff_models(imgs, img_labels)

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
