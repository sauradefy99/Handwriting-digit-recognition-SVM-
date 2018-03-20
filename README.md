
Machine learning semester project for the [Statistical Learning](https://qa.auth.gr/el/class/1/600011290/M1) course at Aristotle University of Thessaloniki.
Task of the project was to perform machine learning algorithms on the benchmark dataset of MNIST, in order to recognize handwritten digit images. 
MNIST was introduced by [Yann LeCunn](http://yann.lecun.com/exdb/mnist/), and contains 70.000 images of 28x28 pixels each, extending our feature vector 
to 784 dimensions. The training set comprises of the first 60.000 images and the testing set of the last 10.000 images. 

I performed classification, clustering, dimensionality reduction and embedding. At best, SVM achieved an 1.8% error rate.

####Dependencies 
* Python 2.7+
* Scikit-learn
* Matplotlib
* Numpy


 
####Classification
By running  ```svm_mnist.py``` we run the SVM classification code.
The code first loads the dataset via its helper function provided by sklearn. 
Then it normalizes each pixel at [0,1].

```
X_train, y_train = np.float32(mnist.data[:60000])/ 255., np.float32(mnist.target[:60000])
```
In order to be able to run this task in a regular machine, we reduce the dimensions from 784 to 90 with PCA. That way, we keep around
91% of the initial information.
![PICTURE](https://github.com/sdimi/handwritten-digits-recognition/blob/master/figures/variance%20explained.png)

After dimensionality reduction, we perform SVM with various kernels and hyperparameters. The following accuracy results are obtained after
5-fold cross validation. 
![PICTURE](https://github.com/sdimi/handwritten-digits-recognition/blob/master/figures/svm%20results.PNG)

Some correct and false classification examples are shown below. At MNIST the "9" digit is confused with "4" sometimes.

| Correct prediction  | False prediction |
| ------------- | ------------- |
| ![PICTURE](https://github.com/sdimi/handwritten-digits-recognition/blob/master/figures/correct%20prediction%20mnist.png)  | ![PICTURE](https://github.com/sdimi/handwritten-digits-recognition/blob/master/figures/false%20prediction%20mnist.png)  |

####Dimensionality Reduction
By running  ```kpca_mnist.py``` we run the lda + kernelPCA code. With the new reduced dimensions, we perform kNN and NearestCentroid.
Please note that kPCA is a memory intensive process, so we limit our training set to 15.000 samples. The following table presents the 
classification accuracy with our eventually reduced dimensions down to 9.
![PICTURE](https://github.com/sdimi/handwritten-digits-recognition/blob/master/figures/mnist%20dim%20reduction.PNG)

####Embedding Projections & Clustering
Finally, we run  ```cluster_mnist.py``` in order to project our dataset in the two-dimensional space, leveraging Spectral and
Isomap embeddings. By keeping 5000 samples for visualization, we perform spectral clustering. To evaluate the 
clustering effectiveness, we compute the cluster completeness score which is under 0.5 for both cases.
The following scatterplots display the embeddings.

| Isomap  | Spectral |
| ------------- | ------------- |
| ![PICTURE](https://github.com/sdimi/handwritten-digits-recognition/blob/master/figures/MNIST%20Isomap.png)  | ![PICTURE](https://github.com/sdimi/handwritten-digits-recognition/blob/master/figures/MNIST%20Spectral.png) |
