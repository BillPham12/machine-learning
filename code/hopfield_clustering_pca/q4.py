import struct
from sklearn.model_selection import KFold
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
from sklearn.datasets import fetch_lfw_people

# load data
dataset = fetch_lfw_people(data_home='.cache/', min_faces_per_person=70)

#convert data
data = dataset['data']
target = dataset['target']
label = []
for index in target:
    m = np.zeros(7)
    m[index] = 1
    label.append(m)
data = np.array(data)
label = np.array(label)


def init_weights(shape,x):
    return tf.Variable(tf.random_normal(shape, stddev=x))

# network architecture
def model(X,standard, pca):
    if not pca:
        w_h1 = init_weights([2914, 625],standard) #  variables
        first_layer = tf.nn.sigmoid(tf.matmul(X,w_h1))
        w_h2 = init_weights([625, 300],standard) #  variables
        second_layer = tf.nn.sigmoid(tf.matmul(first_layer,w_h2))
        w_h3 = init_weights([300, 7],standard) # variables
        return tf.matmul(second_layer,w_h3)
    else:
        w_h1 = init_weights([75, 625],standard) #  variables
        first_layer = tf.nn.sigmoid(tf.matmul(X,w_h1))
        w_h2 = init_weights([625, 300],standard) #  variables
        second_layer = tf.nn.sigmoid(tf.matmul(first_layer,w_h2))
        w_h3 = init_weights([300, 7],standard) # variables
        return tf.matmul(second_layer,w_h3)


epochs = 100
bath_size  = 10
k_fold = KFold(n_splits=10)
non_pca = []
print("Non-PCA")
# place holders
X = tf.placeholder(shape =[None,2914], dtype = tf.float32)
Y = tf.placeholder(shape =[None,7], dtype = tf.float32)
predict_data = model(X,0.6,False)
learning_rate = 0.01
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict_data, labels = Y)) # compute costs
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # construct an optimizer
predict_op = tf.argmax(predict_data, 1)
correct_y = tf.argmax(Y, 1)

for train_index, test_index in k_fold.split(data):
    # splitting data
    trX, teX = data[train_index], data[test_index]
    trY, teY = label[train_index], label[test_index]
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(0,epochs):
            # train non-noise data
            for start, end in zip(range(0, len(trX), bath_size), range(bath_size, len(trX)+1, 1)):
                sess.run(optimizer, feed_dict={X: trX[start:end], Y: trY[start:end]})
            test = sess.run(predict_op, feed_dict = {X:teX})
    non_pca.append(np.mean(np.argmax(teY, axis=1) == test))
    print(np.mean(np.argmax(teY, axis=1) == test))
print("Average accuracy of non_pca", np.mean(non_pca))
pca = []

# transform data with # components 75
component = 75
pca_data = PCA(n_components = component,svd_solver='randomized', whiten=True).fit_transform(data)


print("USING PCA")
X = tf.placeholder(shape =[None,component], dtype = tf.float32)
Y = tf.placeholder(shape =[None,7], dtype = tf.float32)
predict_data = model(X,0.01,True)
learning_rate = 0.01
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict_data, labels = Y)) # compute costs
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # construct an optimizer
predict_op = tf.argmax(predict_data, 1)
correct_y = tf.argmax(Y, 1)

for train_index, test_index in k_fold.split(pca_data):
    # splitting data
    trX, teX = pca_data[train_index], pca_data[test_index]
    trY, teY = label[train_index], label[test_index]
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(0,epochs):
            # train non-noise data
            for start, end in zip(range(0, len(trX), bath_size), range(bath_size, len(trX)+1, 1)):
                sess.run(optimizer, feed_dict={X: trX[start:end], Y: trY[start:end]})
            test = sess.run(predict_op, feed_dict = {X:teX})
    pca.append(np.mean(np.argmax(teY, axis=1) == test))
    print(np.mean(np.argmax(teY, axis=1) == test))
print("Average accuracy of non_pca", np.mean(pca))
list = [i for i in range(1,11)]
plt.plot(list,non_pca,color = "green",label = "non_pca")
plt.plot(list,pca,color = "red",label = "pca")
plt.legend()
plt.title("COMPARISION BETWEEN:Non_PCA and PCA")
plt.xlabel('K steps in K fold', fontsize=10)
plt.ylabel('accuracy percentage', fontsize=10)
plt.show()
