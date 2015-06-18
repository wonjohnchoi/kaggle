import pandas as pd
from sklearn import svm, metrics
from sklearn.decomposition import PCA
# f-score: 0.85 for n_train=2500, n_cross=2500, gamma=0.001
# f-score: 0.88 for n_train=5000, n_cross=2500, gamma=0.001
# f-score: 0.88 for n_train=5000, n_cross=5000, gamma=0.001
# f-score: 0.95 for n_train=2000, n_cross=2500, gamma=0
# f-score: 0.95 for n_train=4000, n_cross=2500, gamma=0
# out of memory with 1GB ram for n_train=7500
# score from kaggle (https://www.kaggle.com/wonjohnchoi/digit-recognizer/pca-svm)
# 0.91757 for n_train=all, n_components=30, gamma=0.001
# 0.97886 for n_train=all, n_components=30, gamma=0
print 'Input train, cross, test'
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
n_train = 4000
n_cross = 2500
train_x = train.ix[:n_train,1:].values.astype('uint8')
train_y = train.ix[:n_train,0].values.astype('uint8')
cross_x = train.ix[n_train:n_train+n_cross,1:].values.astype('uint8')
cross_y = train.ix[n_train:n_train+n_cross,0].values.astype('uint8')
n_test = 5000
test_x = test.ix[:n_test,:].values.astype('uint8')

print 'PCA reduction'
pca = PCA(n_components=40, whiten=True)
pca.fit(train_x)
train_x = pca.transform(train_x)
cross_x = pca.transform(cross_x)
test_x = pca.transform(test_x)

# print train_y
# print type(train_x), type(train_y)
# print len(train_y), len(train_x)
# print train_x[0], train_y[0]

print 'SVM classifier'
clf = svm.SVC()
clf.fit(train_x, train_y)

print '2-fold cross validation'
expected = cross_y;
predicted = clf.predict(cross_x)
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

print 'Output'
test_y = clf.predict(test_x)
pd.DataFrame({"ImageId": range(1,len(test_y)+1), "Label": test_y}).to_csv('out.csv', index=False, header=True)
