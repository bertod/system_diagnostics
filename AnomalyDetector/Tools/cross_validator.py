from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class CrossValidator:

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def split_train_test(self, test_size, random_state=0):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y,
                                                            test_size=test_size,
                                                            random_state=random_state)
        return x_train, x_test, y_train, y_test

    def k_fold(self):
        pass

    def assess_classifier(self, classifier, y_pred, y_test):
        # print(classifier.score(y_test, y_pred))
        # print(confusion_matrix(y_test, y_pred))
        # print(classification_report(y_test, y_pred))
        print('accuracy: ', accuracy_score(y_test, y_pred))
        accuracy = accuracy_score(y_test, y_pred)
        print('f1 measure (weighted): ', f1_score(y_test, y_pred, average='weighted'))
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        print('f1 measure (none): ', f1_score(y_test, y_pred, average=None))
        f1_none = f1_score(y_test, y_pred, average=None)

        print('f1 measure (binary): ', f1_score(y_test, y_pred, average='binary'))
        f1_binary = f1_score(y_test, y_pred, average='binary')

    def assess_kfold(self, classifier, x, y):
        scores = cross_val_score(classifier, x, y, cv=5, scoring='f1_none')
