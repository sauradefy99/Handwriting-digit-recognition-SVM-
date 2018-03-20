
# Author: Souradeep Chakraborty <https://github.com/sauradefy99>


import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn import decomposition
from sklearn import cross_validation
import time # computation time benchmark
from sklearn.lda import LDA
from sklearn import neighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.decomposition import  KernelPCA
from sklearn.utils import shuffle
from sklearn.neighbors.nearest_centroid import NearestCentroid
# Fetching the dataset from 
# mldata.org/repository/data/viewslug/mnist-original through the sklearn helper
mnist = fetch_mldata("MNIST original")

print("------------Data shape (samples, dimensions)--")
print(mnist.data.shape) #(70000, 784)

print("------------Number of classes-----------------")
print(np.unique(mnist.target)) #array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])



# Keeping 60k out of 70k as train-set, as per the creators' instructions.
# Normalizing each pixel at [0,1].


X, y = np.float32(mnist.data[:70000])/ 255., np.float32(mnist.target[:70000])
X, y = shuffle(X,y)
#keeping just 15k training samples due to kPCA memory requirements
X_train, y_train = np.float32(X[:15000])/255., np.float32(y[:15000])
X_test, y_test = np.float32(X[60000:])/ 255., np.float32(y[60000:])

print (mnist.data.shape)
#kernel PCA keeping 300 components
kpca = KernelPCA(kernel="rbf",n_components=300 , gamma=1)
X_kpca = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)
print (kpca)
print(X_kpca.shape)





#lda for dimensionality reduction. It should keep [classes-1] components.
lda = LDA()
print (lda)

X_lda = lda.fit_transform(X_kpca,y_train)
X_test = lda.transform(X_test)
print(X_lda.shape)



#kNN classification
start = int(round(time.time() * 1000))
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(X_lda, y_train)

print (clf)

print("---------(5) Cross validation accuracy--------")
print(cross_validation.cross_val_score(clf, X_lda,y_train, cv=5))


end = int(round(time.time() * 1000))
print("--NN fitting finished in ", (end-start), "ms--------------")


print("---------Test-set dimensions after PCA--------")
print(X_test.shape)

expected = y_test
predicted = clf.predict(X_test)

print("--------------------Results-------------------")
print("Classification report for kNN classifier %s:\n%s\n"
     % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))







#Nearest Centroid classification
start = int(round(time.time() * 1000))


classifier = NearestCentroid()
classifier.fit(X_lda, y_train)
NearestCentroid(metric='euclidean', shrink_threshold=None)
print (classifier)



print("---------(5) Cross validation accuracy--------")
print(cross_validation.cross_val_score(classifier, X_lda,y_train, cv=5))


end = int(round(time.time() * 1000))
print("--Centroid fitting finished in ", (end-start), "ms--------------")


print("---------Test-set dimensions after PCA--------")
print(X_test.shape)

expected = y_test
predicted = classifier.predict(X_test)

print("--------------------Results-------------------")
print("Classification report for Centroid classifier %s:\n%s\n"
     % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


# Code runs in about 6mins at a dual core machine.

