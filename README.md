I undertook this project during the winter vacations of the first year of my college. This project aims at predicting handwritten digits gathered from the MNIST dataset which has 70,000, 28x28 pixels.
I took 60,000 training cases and 10,000 test cases.

I performed classification, clustering, dimensionality reduction and embedding. At best, SVM achieved an 1.8% error rate.

 
1. CLASSIFICATION:

By running  ```svm_mnist.py``` we run the SVM classification code.
The code first loads the dataset via its helper function provided by sklearn. 
Then it normalizes each pixel at [0,1].

```
X_train, y_train = np.float32(mnist.data[:60000])/ 255., np.float32(mnist.target[:60000])
```
In order to be able to run this task in a regular machine, we reduce the dimensions from 784 to 90 with PCA. That way, we keep around
91% of the initial information.

After dimensionality reduction, we perform SVM with various kernels and hyperparameters. The following accuracy results are obtained after
5-fold cross validation. 






2. REDUCING DIMENSIONS:

By running  ```kpca_mnist.py``` we run the lda + kernelPCA code. With the new reduced dimensions, we perform kNN and NearestCentroid.
Please note that kPCA is a memory intensive process, so we limit our training set to 15.000 samples.


3. EMBEDDING PROJECTIONS AND CLUSTERING:

Finally, we run  ```cluster_mnist.py``` in order to project our dataset in the two-dimensional space, leveraging Spectral and
Isomap embeddings. By keeping 5000 samples for visualization, we perform spectral clustering. To evaluate the 
clustering effectiveness, we compute the cluster completeness score which is under 0.5 for both cases.

