import struct
import numpy as np
from matplotlib import pyplot as plt
import random
import tensorflow as tf
import math
import time
from tensorflow.examples.tutorials.mnist import input_data
import sklearn
import sklearn.cluster as sk
from sklearn.utils import shuffle


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

k_list = [50,60,70,80,90,100]
find_best = False
fit_data = np.append(trX, teX, axis=0)
results = []
if find_best:
    for k in k_list:
        kmean = sk.KMeans(n_clusters=k, n_init=1).fit(fit_data)
        clusters = kmean.cluster_centers_
        correct_matrix = []

        index = 0
        for cluster in clusters:
            label = 9999
            best = 9999999
            x = 0
            for i in range(0,len(trX)):
                distance = np.linalg.norm(trX[i] - cluster)
                if distance < best:
                    best = distance
                    x = i
            correct_matrix.append((trY[x],index))
            index += 1
        predict = kmean.predict(teX)
        error = 0
        for i in range(0,len(predict)):
            for correct in correct_matrix:
                if predict[i] == correct[1] and np.array_equal(np.array(teY[i]),np.array(correct[0])) == False:
                    error += 1
        print("Error rate", error/len(teX)*100)
        results.append(error/len(teX)*100)
    plt.plot(k_list,results,color = "green",label = "Error rate")
    plt.legend()
    plt.title("The analysis of hopfield network")
    plt.xlabel('Number of clusters', fontsize=10)
    plt.ylabel('The error rate', fontsize=10)
    plt.show()

np.random.seed(1)
def init_weights(shape,x):
    return tf.Variable(tf.random_normal(shape, stddev=x))

num_of_layers = tf.constant(100,dtype = tf.int32)
# 2 place holders
X = tf.placeholder(shape =[None,784], dtype = tf.float32)
Y = tf.placeholder(shape = [None,10],dtype = tf.float32)

# activation function
# this part is tricky
def rbf(x, clusters,beta):
    pros = []
    x1 = tf.to_float(tf.tile(tf.expand_dims(x, 1), [1, len(clusters), 1]))
    out= tf.to_float(tf.reshape(tf.tile(clusters, [tf.shape(x)[0], 1]), [tf.shape(x)[0],len(clusters),784]))
    k = tf.square(tf.norm(tf.subtract(x1, out), axis=-1))
    out = tf.exp(tf.multiply(tf.negative(beta),k))
    return out

def beta(clusters, data):
    sig = np.zeros([len(clusters)], dtype=np.float32)
    n = np.zeros([len(clusters)], dtype=np.float32)
    for x in data:
        # our goal is finding the best result and its index
        # then store the result into sigma array
        # n is used to store the total number of elements stored in a specific
        # cluster
        best = 99999
        index = -1
        for j in range(0, len(clusters)):
            d = np.linalg.norm(x - clusters[j])
            if d < best:
                best = d
                index = j
        sig[index] += best
        n[index] += 1
    for i in range(0,len(sig)):
        sig[i]  = sig[i]/n[i]
    sig = np.array(sig)
    return tf.divide(1, tf.multiply(2., tf.square(sig)))

# network architecture
def model(X,center,layer,standard,beta):
    w_h1 = init_weights([784, layer],standard) #  variables
    first_layer = rbf(X,center,beta)
    w_o = init_weights([layer, 10],standard)
    return tf.matmul(first_layer,w_o)


layers = [50,60,70,75,80,85,90,100]

# combine train and test data
data = np.append(trX, teX, axis=0)
labels = np.append(trY, teY, axis=0)
# shuffle both train and test data
data,labels = shuffle(data, labels)
# implement k fold
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=5)

results = []
for layer in layers:
    print("The current number of layer",layer)
    kmean = sk.KMeans(n_clusters=layer, n_init=1).fit(fit_data)
    centers = kmean.cluster_centers_
    # calculate the beta from the result of k means
    #->to compute the radius
    b = beta(centers,fit_data)
    # the model
    predict_data = model(X,centers,layer,0.1,b)
    # the cost is similar to the previous assignment
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict_data, labels=Y)) # compute cost
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
    predict_op = tf.argmax(predict_data, 1)
    # accurary fomular
    true_y = tf.argmax(Y,1)
    # K fold cross correlation
    sum = []
    epochs = 1
    for train_index, test_index in k_fold.split(data):
        # splitting data
        trX, teX = data[train_index], data[test_index]
        trY, teY = labels[train_index], labels[test_index]

        # Launch the graph in a session
        batch_size = 128
        with tf.Session() as sess:
            # initilize variables
            tf.global_variables_initializer().run()
            for i in range(0,epochs):
                for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
                    sess.run(predict_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
                correctness = np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX}))
            sum.append(correctness)
            print("The current accurary", sum[len(sum) - 1])
    # adding the mean as the result of k fold analysis
    results.append(np.mean(sum))
    print("The aggregate of accuracy:", np.mean(sum))


plt.plot(layers,results,color = "green",label = "accuracy rate")
plt.legend()
plt.title("The analysis of hopfield network")
plt.xlabel('Number of clusters', fontsize=10)
plt.ylabel('The accuracy rate of 1 epoch in hopfield network', fontsize=10)
plt.show()
