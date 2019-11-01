import os
from PIL import Image
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection
from sklearn.metrics import classification_report
from feature import NPDFeature
from ensemble import AdaBoostClassifier


def get_graypic(path, x, y):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        # initialize label
        label = None

        # categorize the dataset into two parts(with label 1 or -1)
        if path == 'datasets/original/face/':
            label = 1
        elif path == 'datasets/original/nonface/':
            label = -1

        # transform initial images into graypic with size 24 x 24
        image = Image.open(file_path).convert('L').resize((24, 24))

        # extract NPD features in images
        if (x is None) & (y is None):
            x = np.array([NPDFeature(np.asarray(image)).extract()])
            y = np.array([label])
        else:
            x = np.vstack((x, NPDFeature(np.asarray(image)).extract()))
            y = np.vstack((y, label))
    return x, y


if __name__ == "__main__":
    gray_x, gray_y = get_graypic('datasets/original/face/', None, None)
    gray_x, gray_y = get_graypic('datasets/original/nonface/', gray_x, gray_y)
    # print(gray_x.shape) //(1000,165600)
    # print(gray_y.shape) //(1000,1)
    AdaBoostClassifier.save(gray_x, 'features')
    AdaBoostClassifier.save(gray_y, 'labels')

    features = AdaBoostClassifier.load('features')
    label = AdaBoostClassifier.load('labels')

    # split the dataset with 80% for training set and 20% for testing set
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(features, label, test_size=0.3)
    # train the base learners to form a boosted classifier
    adaboostclassifier = AdaBoostClassifier(DecisionTreeClassifier(criterion="entropy",
                                                                   max_depth=2,
                                                                   class_weight="balanced"), 15).fit(X_train, Y_train)
    # use boosted classifier generated above to make predictions
    predict_y = adaboostclassifier.predict(X_test)

    # save the precision, recall, f1-score information and so on to evaluate the model
    with open('classifier_report.txt', "wb") as f:
        report = classification_report(Y_test, predict_y, target_names=["face", "nonface"])
        f.write(report.encode())
