import pandas as pd
from sklearn import svm, metrics
train = pd.read_csv('data/train.csv')
#test = pd.read_csv('data/test.csv')
train_x = train.ix[:,1:].values.astype('int32')
train_y = train.ix[:,0].values.astype('int32')
n_train = len(train_y)
# print train_y
print len(train_y), len(train_x)
print train_x[0], train_y[0]
clf = svm.SVC(gamma=0.001)
clf.fit(train_x[:n_train / 2], train_y[:n_train / 2])
expected = train_y[n_train / 2:]
predicted = clf.predict(train_x[n_train / 2:])
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
