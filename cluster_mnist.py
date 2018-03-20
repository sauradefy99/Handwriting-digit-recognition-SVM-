
# Author: Souradeep Chakraborty <https://github.com/sauradefy99>


import numpy as np
#importing dependencies
from sklearn.datasets import fetch_mldata
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition)
import numpy as np
import time
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.utils import shuffle
from matplotlib import offsetbox
# Fetching the dataset from 
# mldata.org/repository/data/viewslug/mnist-original through the sklearn helper
mnist = fetch_mldata("MNIST original")

print("------------Data shape (samples, dimensions)--")
print(mnist.data.shape) #(70000, 784)

print("------------Number of classes-----------------")
print(np.unique(mnist.target)) #array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])





X, y = np.float32(mnist.data[:70000])/ 255., np.float32(mnist.target[:70000])
X, y = shuffle(X,y)

X_train, y_train = np.float32(X[:5000])/255., np.float32(y[:5000])
X_test, y_test = np.float32(X[60000:])/ 255., np.float32(y[60000:])

print (mnist.data.shape)





# Spectral embedding projection 
print("Computing Spectral embedding")
start = int(round(time.time() * 1000))
X_spec = manifold.SpectralEmbedding(n_components=2, affinity='nearest_neighbors', gamma=None, random_state=None, eigen_solver=None, n_neighbors=5).fit_transform(X_train)
end = int(round(time.time() * 1000))
print("--Spectral Embedding finished in ", (end-start), "ms--------------")
print("Done.")


# Isomap projection 
#print("Computing Isomap embedding")
#start = int(round(time.time() * 1000))
#X_iso = manifold.Isomap(n_neighbors=5, n_components=2).fit_transform(X_train)
#end = int(round(time.time() * 1000))
#print("--Isomap finished in ", (end-start), "ms--------------")
#print("Done.")


#spectral clustering, fitting and predictions
spectral = cluster.SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="nearest_neighbors")

#X = spectral.fit(X_iso)
X = spectral.fit(X_spec)

#y_pred = spectral.fit_predict(X_iso)
y_pred = spectral.fit_predict(X_spec)


# clustering evaluation metrics
print(confusion_matrix(y_train, y_pred))
print (completeness_score(y_train, y_pred))

 

with plt.style.context('fivethirtyeight'):       
	plt.title("Spectral embedding & spectral clustering on MNIST")
	plt.scatter(X_spec[:, 0], X_spec[:, 1], c=y_pred, s=50, cmap=plt.cm.get_cmap("jet", 10))
	plt.colorbar(ticks=range(10))
	plt.clim(-0.5, 9.5)
plt.show()


# Code runs in about 6mins at a dual core machine.

