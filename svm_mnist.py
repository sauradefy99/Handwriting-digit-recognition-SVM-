
# Author: Souradeep Chakraborty <https://github.com/sauradefy99>


import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn import decomposition
from sklearn import cross_validation
import time # computation time benchmark

# Fetching the dataset from 
# mldata.org/repository/data/viewslug/mnist-original through the sklearn helper
mnist = fetch_mldata("MNIST original")

print("------------Data shape (samples, dimensions)--")
print(mnist.data.shape) #(70000, 784)

print("------------Number of classes-----------------")
print(np.unique(mnist.target)) #array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
print("------------A glimpse of the data--------------")
print(mnist.data)

# Keeping 60k out of 70k as train-set, as per the creators' instructions.
# Normalizing each pixel at [0,1].
X_train, y_train = np.float32(mnist.data[:60000])/ 255., np.float32(mnist.target[:60000])

# Fitting the train-set in PCA in order to reduce dimensions.
# N_components set after experimenting with variance, trying to keep 
# over 90% of the initial information.
pca = RandomizedPCA(n_components=90)
pca.fit(X_train)

# Plot explained variance ratio for the new dimensions.
print("---------Variance explained-------------------")
print(np.sum(pca.explained_variance_ratio_))

#Styling graph with thicker lines and flat colors
with plt.style.context('fivethirtyeight'):    
    plt.show()
    plt.xlabel("Principal components ")
    plt.ylabel("Variance")
    plt.plot(pca.explained_variance_ratio_)
    plt.title('Variance Explained by Extracted Componenent')

plt.show()

# Fitting the new dimensions to the train-set.
train_ext = pca.fit_transform(X_train)

# New dimensions
print("---------Train-set dimensions after PCA--------")
print(train_ext.shape)


# Check how much time it takes to fit the SVM
start = int(round(time.time() * 1000))


# Fitting training data to SVM classifier.
# Fine-tuning parameters session included
# rbf, poly, linear and different values of gammma and C
classifier = svm.SVC(gamma=0.01, C=3, kernel='rbf')
classifier.fit(train_ext,y_train)

print("---------(5) Cross validation accuracy--------")
print(cross_validation.cross_val_score(classifier, train_ext, y_train, cv=5))

# End of time benchmark
end = int(round(time.time() * 1000))
print("--SVM fitting finished in ", (end-start), "ms--------------")

# Using the last 10k samples as test set against the already trained cross-validated train-set.
X_test, y_test = np.float32(mnist.data[60000:]) / 255., np.float32(mnist.target[60000:])

# Fitting the new dimensions.
test_ext = pca.transform(X_test)
print("---------Test-set dimensions after PCA--------")
print(test_ext.shape)
expected = y_test
predicted = classifier.predict(test_ext)

print("--------------------Results-------------------")
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


# Selecting randomly some testing samples
# Comparing expected label with predicted 
# Falsely classified images appeared after some experimentation with "size" and 
# by lowering the train-set size in order to decrease overall accuracy
for i in np.random.choice(np.arange(0, len(expected)), size = (3,)):
    pred = classifier.predict(np.atleast_2d(test_ext[i]))	
    image = (X_test[i] * 255).reshape((28, 28)).astype("uint8")	
    plt.figure()  
    plt.axis("off")
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Actual digit is {0}, predicted {1}".format(expected[i], pred[0]))

plt.show()

# Code runs in about 6mins at a dual core machine.



