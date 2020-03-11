from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold


class CrossValidator:
    """
        x = pd.DataFrame
        y = pd.Series
    """
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def split_train_test(self, test_size, random_state=0):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y,
                                                            test_size=test_size,
                                                            random_state=random_state)
        return x_train, x_test, y_train, y_test

    def k_fold(self):
        kf = KFold(n_splits=5, shuffle=True)
        # kf = StratifiedKFold(n_splits=10, shuffle=True)
        # for train_index, test_index in kf.split(self.x, self.y.to_list()):
        for train_index, test_index in kf.split(self.x):
            yield self.x.iloc[train_index], self.x.iloc[test_index], \
                  self.y.iloc[train_index], self.y.iloc[test_index]

    def assess_classifier(self, classifier, y_pred, y_test):

        print('accuracy: ', accuracy_score(y_test, y_pred))
        accuracy = accuracy_score(y_test, y_pred)
        print('f1 measure (weighted): ', f1_score(y_test, y_pred, average='weighted'))
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        print('f1 measure (none): ', f1_score(y_test, y_pred, average=None))
        f1_none = f1_score(y_test, y_pred, average=None)

        print('f1 measure (binary): ', f1_score(y_test, y_pred, pos_label=1, average='binary'))
        f1_binary = f1_score(y_test, y_pred, average='binary')
        return f1_binary
