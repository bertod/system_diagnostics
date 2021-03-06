from sklearn.svm import SVC
from AnomalyDetector.Tools.cross_validator import CrossValidator
import statistics
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import sys
from sklearn.model_selection import KFold, StratifiedKFold


class ClassifierModeler:
    """
    Use this class for apply classification algorithm to your data.
    Add a method for each algorithm you want to implement
    """

    def __init__(self, x, y, algorithm="svm", validation="train_test_split"):
        self.x = x
        self.y = y
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.algorithm = algorithm
        self.classifier = None
        self.validation = validation
        self.y_pred = None
        self.validator = CrossValidator(self.x, self.y)

    def train_svm(self):
        self.classifier = SVC(kernel='rbf', C=1)
        # self.classifier = SVC(kernel='rbf', C=1, class_weight="balanced")
        self.classifier.fit(self.x_train, self.y_train)

    def predict_svm(self, x_test_optional=None, y_test_optional=None):
        if not x_test_optional is None and not y_test_optional is None:
            self.x_test = x_test_optional
            self.y_test = y_test_optional
        if not y_test_optional is None:
            self.y_test = y_test_optional
            # x_test = x_test_optional
        # else:
            # x_test = self.x_test
        self.y_pred = self.classifier.predict(self.x_test)

    def run(self, x_test_optional=None, y_test_optional=None):
        if self.algorithm == "svm":
            if self.validation == "train_test_split":
                self.x_train, self.x_test, self.y_train, self.y_test = \
                    self.validator.split_train_test(test_size=0.3, random_state=0)
                self.train_svm()
                self.predict_svm(x_test_optional, y_test_optional)
                self.assess()
                self.print_classifier()
                return self.y_pred, self.x_test

            elif self.validation == "kfolds":
                f1_list = []
                f1_list2 = []
                foldn = 1
                kf = KFold(n_splits=5, shuffle=False)

                for xtrain, xtest, ytrain, ytest in self.validator.k_fold():
                    """for train_index, test_index in kf.split(self.x):
                        xtrain = self.x.iloc[train_index]
                        xtest = self.x.iloc[test_index]
                        ytrain = self.y.iloc[train_index]
                        ytest = self.y.iloc[test_index]"""

                    print("---------- FOLD n: ", foldn, ' ---------------')
                    self.x_train = xtrain
                    self.x_test = xtest
                    self.y_train = ytrain
                    self.y_test = ytest
                    self.train_svm()
                    if(y_test_optional is not None):
                        y_test_optional_ = y_test_optional[ytest.index]
                    else:
                        y_test_optional_ = None
                    #y_test_optional_ = y_test_optional.iloc[test_index]
                    self.predict_svm(None, y_test_optional_)
                    f1_list.append(self.assess())
                    if foldn == 5:
                        self.print_classifier()
                    foldn += 1
                    classified_series = pd.DataFrame({"predicted": self.y_pred, "generate truth": self.y_test, "actual truth": y_test_optional_},
                                                     index=self.x_test.index)
                    print(classified_series)

                print("avg f1 score from kfolds: ", statistics.mean(f1_list))
                return None, None

            elif self.validation == "none":
                self.x_train = self.x
                self.y_train = self.y
                self.train_svm()
                #if not x_test_optional is None and not y_test_optional is None:
                    #self.predict_svm(x_test_optional, y_test_optional)


    def assess(self):
        return self.validator.assess_classifier(self.classifier, self.y_pred, self.y_test)

    def print_classifier(self):
        h = .02  # step size in the mesh
        x_plot = self.x_test
        # x_plot = self.x_train
        y_plot = self.y_test
        # y_plot = self.y_train

        fig = plt.figure(figsize=(10, 10))
        # create a mesh to plot in
        x_min, x_max = x_plot.iloc[:, 0].min() - 1, x_plot.iloc[:, 0].max() + 1
        y_min, y_max = x_plot.iloc[:, 1].min() - 1, x_plot.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # title for the plots
        # title = 'SVC without kernel'
        title = 'SVC with RBF kernel'
        Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(x_plot.iloc[:, 0], x_plot.iloc[:, 1], c=y_plot.to_list(), cmap=plt.cm.coolwarm)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(title)
        plt.show()

    def print_classifier2(self):
        h = .02  # step size in the mesh
        x_plot = self.x_test
        # x_plot = self.x_train
        y_plot = self.y_test
        # y_plot = self.y_train

        fig = plt.figure(figsize=(10, 10))
        # create a mesh to plot in
        x_min, x_max = x_plot.iloc[:, 0].min() - 1, x_plot.iloc[:, 0].max() + 1
        y_min, y_max = x_plot.iloc[:, 1].min() - 1, x_plot.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # title for the plots
        # title = 'SVC without kernel'
        title = 'SVC with Linear kernel'
        Z = self.classifier2.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(x_plot.iloc[:, 0], x_plot.iloc[:, 1], c=y_plot.to_list(), cmap=plt.cm.coolwarm)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(title)
        plt.show()