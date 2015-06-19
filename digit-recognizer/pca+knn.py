# f-1 score: 0.91 for n_train=3000, n_cross=2500, without pca
# f-1 score: 0.91 for n_train=3000, n_cross=2500, with pca
# out of memory with 1GB ram for n_train=7500
# score from kaggle (https://www.kaggle.com/wonjohnchoi/digit-recognizer/knn)
# 0.96986 for n_train=all, with pca

import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

import time
start = time.time()
def elapsed(): return time.time() - start

n_train = 3000
n_cross = 2500
n_test = 5000
print('Input train(%d), cross(%d), test(%d) at %ds' % (n_train, n_cross, n_test, elapsed()))
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train_x = train.ix[:n_train,1:].values.astype('uint8')
train_y = train.ix[:n_train,0].values.astype('uint8')
cross_x = train.ix[n_train:n_train+n_cross,1:].values.astype('uint8')
cross_y = train.ix[n_train:n_train+n_cross,0].values.astype('uint8')
test_x = test.ix[:n_test,:].values.astype('uint8')

# Interesting to note: for SVM, PCA was used to increase performance
# for KNN, PCA does not affect performance but it reduces the running time of script by one third.
print('PCA reduction at %ds' % elapsed())
pca = PCA(n_components=36, whiten=True)
pca.fit(train_x)
train_x = pca.transform(train_x)
cross_x = pca.transform(cross_x)
test_x = pca.transform(test_x)

print('KNN classifier at %ds' % elapsed())
clf = KNeighborsClassifier()
clf.fit(train_x, train_y)

print('2-fold cross validation at %ds' % elapsed())
expected = cross_y;
predicted = clf.predict(cross_x)
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

print('Output at %ds' % elapsed())
test_y = clf.predict(test_x)
pd.DataFrame({"ImageId": range(1,len(test_y)+1), "Label": test_y}).to_csv('out.csv', index=False, header=True)

print('Exit at %ds' % elapsed())