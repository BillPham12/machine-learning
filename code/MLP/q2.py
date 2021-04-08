from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import math
import random
import time
from PIL import Image
import os

def load_data(path):
    images = []
    for img in os.listdir(path):
        img = Image.open(os.path.join(path,img))
        img.load()
        data = np.array(img)
        data.shape = (7,20)
        images.append(data)
    return images

path = "C:/Users/bill/Desktop/COMP4107/Assignment2/q2_data"
#load data
def renew_data(input):
    output = []
    for image in input:
        new_image = []
        for x in image:
            row = []
            for i in range (0,20,4):
                if x[i] == 255 or x[i+1] == 255 or x[i+2] == 255 or x[i+3] == 255:
                    row.append(1)
                else:
                    row.append(0)
            new_image.append(row)
        new_image = np.array(new_image)
        new_image.shape = (1,35)
        output.append(new_image)
    return output

loaded_data = load_data(path)

np.random.seed(1)
# preparing data
def data_noisy_generation(data,noisy):
    test_data = []
    for image in data:
        test_data.append(image)

    for image in test_data:
        bits = []
        index = 0
        for x in image[0]:
            if x == 1:
                bits.append(index)
            index = index + 1
        for y in range(0,noisy):
            index = random.randrange(0,len(bits))
            image[0][bits[index]] = 0
    return test_data

def result_generation():
    output = []
    for x in range(0,31):
        row = []
        row.append([])
        for y in range(0,31):
            if x == y:
                row[0].append(1)
            else:
                row[0].append(0)
        output.append(row)
    return output


# train data
train_data = np.array(renew_data(loaded_data))
#train_data.shape = (31,35)
train_noisy1 = np.array(data_noisy_generation(renew_data(loaded_data),1))
#train_noisy1.shape= (31,35)
train_noisy2 = np.array(data_noisy_generation(renew_data(loaded_data),2))
#train_noisy2.shape= (31,35)
train_noisy3 = np.array(data_noisy_generation(renew_data(loaded_data),3))
#train_noisy3.shape= (31,35)
train_y = np.array(result_generation())
trainX = np.concatenate((train_data,train_noisy3),axis = 0)
#np.random.shuffle(noisy_training_data)
trainY = np.concatenate((train_y,train_y),axis = 0)



def init_weights(shape,x):
    return tf.Variable(tf.random_normal(shape, stddev=x))

# 2 place holders
X = tf.placeholder(shape =[None,35], dtype = tf.float32)
Y = tf.placeholder(shape = [None,31],dtype = tf.float32)

neurons = [5,10,15,20,25]
# model
def model(X,layer,standard):
    w_h1 = init_weights([35, layer],standard) #  variables
    b1 = init_weights([layer],standard) #  variables
    w_h2 = init_weights([layer, 31],standard) # variables
    first_layer = tf.nn.sigmoid(tf.matmul(X,w_h1)+ b1)
    return tf.matmul(first_layer,w_h2)


train_noisy3.shape= (31,35)
train_noisy2.shape= (31,35)
train_noisy1.shape= (31,35)
train_data.shape= (31,35)
train_y.shape = (31,31)

part = 'b'
if part == 'a':
    graph = []
    for neu in neurons:
        graph.append([])
        print("Calculating the recognition errors of", neu, "neurons")
        predict_data = model(X,neu,0.01)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict_data, labels = Y)) # compute costs
        optimizer = tf.train.AdamOptimizer(0.03).minimize(cost) # construct an optimizer
        predict_op = tf.argmax(predict_data, 1)
        correct_y = tf.argmax(Y, 1)
        # Calculating the difference between the outcome and the standard output
        # by ccounting the number of correct predictions
        correct_predict = tf.equal(predict_op, correct_y)
        average_correct = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        average_error = 1 - average_correct
        with tf.Session() as sess:
            # you need to initialize all variables
            tf.global_variables_initializer().run()
            for i in range(0,399):
                for x in range(0,len(train_data)):
                    sess.run([cost,optimizer], feed_dict={X: trainX[x], Y: trainY[x]})
            er = sess.run(average_error, feed_dict={X: train_data, Y: train_y})*100
            er1 = sess.run(average_error, feed_dict={X: train_noisy1, Y: train_y})*100
            er2 = sess.run(average_error, feed_dict={X: train_noisy2, Y: train_y})*100
            er3 = sess.run(average_error, feed_dict={X: train_noisy3, Y: train_y})*100
            graph[len(graph)-1].append(er)
            graph[len(graph)-1].append(er1)
            graph[len(graph)-1].append(er2)
            graph[len(graph)-1].append(er3)

    list = [0,1,2,3]
    five_neurons= plt.plot(list,graph[0],color = "blue",label = "5 hidden neurons")
    ten_neurons =plt.plot(list,graph[1],color = "red",label = "10 hidden neurons")
    fifteen_neurons= plt.plot(list,graph[2],color = "green", label = "15 hidden neurons")
    twenty_neurons =plt.plot(list,graph[3],color = "purple", label = "20 hidden neurons")
    twenty_five_neurons = plt.plot(list,graph[3],color = "black", label = "25 hidden neurons")
    plt.legend()
    plt.xlabel('Noisy levels', fontsize=15)
    plt.ylabel('Percentage of recognition errors', fontsize=15)
    plt.show()

elif part == 'b':
    graph = []
    neu = 15
    print("Calculating the recognition errors of", neu, "neurons")
    predict_data = model(X,neu,0.01)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict_data, labels = Y)) # compute costs
    optimizer = tf.train.AdamOptimizer(0.044).minimize(cost) # construct an optimizer
    predict_op = tf.argmax(predict_data, 1)
    correct_y = tf.argmax(Y, 1)
    # Calculating the difference between the outcome and the standard output
    # by ccounting the number of correct predictions
    correct_predict = tf.equal(predict_op, correct_y)
    average_correct = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    average_error = 1 - average_correct

    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        for i in range(0,300):
            train_data.shape = (31,1,35)
            train_y.shape = (31,1,31)
            # train non-noise data
            for x in range(0,len(train_data)):
                sess.run([cost,optimizer], feed_dict={X: train_data[x], Y: train_y[x]})
            train_data.shape = (31,35)
            train_y.shape = (31,31)
            graph.append(sess.run(average_error, feed_dict={X: train_data, Y: train_y}))
        # train non-noise data
        while True:
            train_noisy3.shape = (31,1,35)
            train_y.shape = (31,1,31)
            for x in range(0,len(train_noisy3)):
                sess.run([cost,optimizer], feed_dict={X: train_noisy3[x], Y: train_y[x]})
            train_noisy3.shape = (31,35)
            train_y.shape = (31,31)
            if sess.run(average_error, feed_dict={X: train_noisy3, Y: train_y}) < 0.01:
                break
        # re-train non-noise data
        for i in range(0,20):
            train_data.shape = (31,1,35)
            train_y.shape = (31,1,31)
            for x in range(0,len(train_data)):
                sess.run([cost,optimizer], feed_dict={X: train_data[x], Y: train_y[x]})
            train_data.shape = (31,35)
            train_y.shape = (31,31)
            graph.append(sess.run(average_error, feed_dict={X: train_data, Y: train_y}))
    list = [i for i in range(0,320)]
    plt.plot(list,graph,color = "blue")
    plt.show()

elif part == 'c':
    graph = []
    graph.append([])
    graph.append([])
    neu = 15
    print("Calculating the recognition errors of", neu, "neurons")
    predict_data = model(X,neu,0.01)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict_data, labels = Y)) # compute costs
    optimizer = tf.train.AdamOptimizer(0.03).minimize(cost) # construct an optimizer
    predict_op = tf.argmax(predict_data, 1)
    correct_y = tf.argmax(Y, 1)
    # Calculating the difference between the outcome and the standard output
    # by ccounting the number of correct predictions
    correct_predict = tf.equal(predict_op, correct_y)
    average_correct = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    average_error = 1 - average_correct
    print("Training with noise")
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        for i in range(0,300):
            train_data.shape = (31,1,35)
            train_y.shape = (31,1,31)
            # train non-noise data
            for x in range(0,len(train_data)):
                sess.run([cost,optimizer], feed_dict={X: train_data[x], Y: train_y[x]})
            train_data.shape = (31,35)
            train_y.shape = (31,31)

        # train non-noise data
        while True:
            train_noisy3.shape = (31,1,35)
            train_y.shape = (31,1,31)
            for x in range(0,len(train_noisy3)):
                sess.run([cost,optimizer], feed_dict={X: train_noisy3[x], Y: train_y[x]})
            train_noisy3.shape = (31,35)
            train_y.shape = (31,31)
            if sess.run(average_error, feed_dict={X: train_noisy3, Y: train_y}) < 0.01:
                break
        # re-train non-noise data
        for i in range(0,20):
            train_data.shape = (31,1,35)
            train_y.shape = (31,1,31)
            for x in range(0,len(train_data)):
                sess.run([cost,optimizer], feed_dict={X: train_data[x], Y: train_y[x]})
            train_data.shape = (31,35)
            train_y.shape = (31,31)
            graph.append(sess.run(average_error, feed_dict={X: train_data, Y: train_y}))

        train_data.shape = (31,35)
        train_noisy1.shape = (31,35)
        train_noisy2.shape = (31,35)
        train_noisy3.shape = (31,35)
        train_y.shape = (31,31)
        er = sess.run(average_error, feed_dict={X: train_data, Y: train_y})*100
        er1 = sess.run(average_error, feed_dict={X: train_noisy1, Y: train_y})*100
        er2 = sess.run(average_error, feed_dict={X: train_noisy2, Y: train_y})*100
        er3 = sess.run(average_error, feed_dict={X: train_noisy3, Y: train_y})*100
        graph[0].append(er)
        graph[0].append(er1)
        graph[0].append(er2)
        graph[0].append(er3)
    print("Training without noise")
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        for i in range(0,350):
            # first train non-noise data
            for x in range(0,len(train_data)):
                sess.run([cost,optimizer], feed_dict={X: trainX[x], Y: trainY[x]})
        train_data.shape = (31,35)
        train_y.shape = (31,31)
        er = sess.run(average_error, feed_dict={X: train_data, Y: train_y})*100
        er1 = sess.run(average_error, feed_dict={X: train_noisy1, Y: train_y})*100
        er2 = sess.run(average_error, feed_dict={X: train_noisy2, Y: train_y})*100
        er3 = sess.run(average_error, feed_dict={X: train_noisy3, Y: train_y})*100
        graph[1].append(er)
        graph[1].append(er1)
        graph[1].append(er2)
        graph[1].append(er3)
    list = [0,1,2,3]
    five_neurons= plt.plot(list,graph[0],color = "red",linewidth = 2,label = "trained with noise")
    ten_neurons =plt.plot(list,graph[1],color = "black",linewidth = 3,label = "trained without noise")
    plt.legend()
    plt.xlabel('Noisy levels', fontsize=15)
    plt.ylabel('Percentage of recognition errors', fontsize=15)
    plt.show()
