from sklearn.svm import SVC
from AnomalyDetector.Tools.cross_validator import CrossValidator


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
        self.classifier = SVC(kernel='linear', C=1)
        self.classifier.fit(self.x_train, self.y_train)

    def predict_svm(self, x_test_optional=None):
        if x_test_optional:
            self.x_test = x_test_optional
            # x_test = x_test_optional
        # else:
            # x_test = self.x_test
        self.y_pred = self.classifier.predict(self.x_test)

        # self.y_pred = self.classifier.predict(x_test)
        # return self.y_pred, x_test

    def run(self):
        if self.algorithm == "svm":
            if self.validation == "train_test_split":
                self.x_train, self.x_test, self.y_train, self.y_test = \
                    self.validator.split_train_test(test_size=0.3, random_state=0)
                self.train_svm()
                self.predict_svm()
                return self.y_pred, self.x_test

            elif self.validation == "kfolds":
                self.validator.assess_kfold()
            elif self.validation == "none":
                self.x_train = self.x
                self.y_train = self.y
                self.train_svm()

    def assess(self):
        self.validator.assess_classifier(self.classifier, self.y_pred, self.y_test)



