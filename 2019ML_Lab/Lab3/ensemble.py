import pickle
import numpy as np


class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.limit = n_weakers_limit

        self.weak_classifier_set = [] # the set saves all the weak classifiers
        self.alpha_total = [] # the set saves the alpha of all weak classifiers

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        self.w = np.ones(X.shape[0]) / X.shape[0]  # initialize parameter w

        for i in range(self.limit):
            classifier = self.weak_classifier.fit(X, y, self.w)
            self.weak_classifier_set.append(classifier)

            weak_predict = classifier.predict(X)
            error = np.sum(self.w * (weak_predict != y.reshape(-1, )))

            if error > 0.5:
                break

            alpha = 0.5 * np.log((1 - error)/error)

            self.alpha_total.append(alpha)

            z = np.multiply(self.w, np.exp(-self.alpha_total[i] * np.multiply(y.reshape(-1, ), weak_predict)))

            self.w = z / np.sum(z)

        return self

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        # initialize score
        score = np.zeros(X.shape[0])
        # prediction
        for i in range(self.limit):
            score += self.alpha_total[i]*self.weak_classifier_set[i].predict(X)
        return score

    def predict(self, X, threshold=0):
        '''Predict the catagories for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        # initialize y_predict
        y_predict = np.zeros(X.shape[0])
        score = self.predict_scores(X)
        # categorize into two classes
        y_predict[np.where(score > threshold)] = 1
        y_predict[np.where(score < threshold)] = -1
        return y_predict

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            return pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
