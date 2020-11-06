# COMP9517 PROJECT - INDIVIDUAL PART
# attempt_surf.py train model with SURF
# AUTHOR: Rui Li
# Email: z5202952@ad.unsw.edu.au

# import

# load the images
from ultilities import *

if __name__ == '__main__':
    for n in [1000]:
        print("hessianThreshold: ", n)
        # for seg in [True, False]:
        #     print("segmented: ", seg)
        imgs, img_labels = surf_features(n, segmented=True)
        X_train, X_test, y_train, y_test = train_test_split(np.array(imgs), np.array(img_labels), test_size=0.25,
                                                            shuffle=True, random_state=10)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        print("accuracy:\t", accuracy_score(y_test, predictions))
        print("recall:\t\t", recall_score(y_test, predictions, average='macro'))

        from sklearn.metrics import roc_auc_score

        y_scores = clf.decision_function(X_test)
        print("AUC:", roc_auc_score(y_test, y_scores))

        from sklearn.metrics import f1_score

        print("f1:", f1_score(y_test, predictions, zero_division=1))
