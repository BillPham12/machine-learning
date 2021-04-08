from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import math
import random
import get_data_for_mlp
from tensorflow.python.keras.preprocessing.text import Tokenizer
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

# word frequency
word_frequency = 8000
#get data with # word_frequency
data,labels = get_data_for_mlp.get_data(word_frequency)
# get data vectorized by ajective words
special_data = get_data_for_mlp.get_special_data()

special_data, data, labels = shuffle(special_data, data, labels,random_state = 1)

def init_weights(shape,x):
    return tf.Variable(tf.random_normal(shape, stddev=x))

# 2 place holders
X = tf.placeholder(shape =[None,word_frequency], dtype = tf.float32)
Y = tf.placeholder(shape = [None,2],dtype = tf.float32)

# normal model with 2 hidden layers
def normal_model(X,standard,keep_prob):
    w_h1 = init_weights([word_frequency, 625],standard) #  variables
    b1 = init_weights([625],standard) #  variables
    first_layer = tf.nn.sigmoid(tf.matmul(X,w_h1)+ b1)
    first_layer = tf.nn.dropout(first_layer, keep_prob=keep_prob)

    b2 = init_weights([300],standard) #  variables
    w_h2 = init_weights([625, 300],standard) # variables
    second_layer = tf.nn.sigmoid(tf.matmul(first_layer,w_h2)+ b2)
    second_layer = tf.nn.dropout(second_layer, keep_prob=keep_prob)
    w_h3 = init_weights([300,2],standard) # variables

    return tf.matmul(second_layer,w_h3)


# 2 place holders
J = tf.placeholder(shape =[None,9587], dtype = tf.float32)
# speical model with 2 hidden layers
def special_model(J,standard,keep_prob):
    w_h1 = init_weights([9587, 625],standard) #  variables
    b1 = init_weights([625],standard) #  variables
    first_layer = tf.nn.sigmoid(tf.matmul(J,w_h1)+ b1)
    first_layer = tf.nn.dropout(first_layer, keep_prob=keep_prob)

    b2 = init_weights([300],standard) #  variables
    w_h2 = init_weights([625, 300],standard) # variables
    second_layer = tf.nn.sigmoid(tf.matmul(first_layer,w_h2)+ b2)
    second_layer = tf.nn.dropout(second_layer, keep_prob=keep_prob)
    w_h3 = init_weights([300,2],standard) # variables

    return tf.matmul(second_layer,w_h3)

special_results = []
# model
predict_data = normal_model(X,0.15,0.5)
# cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict_data, labels = Y)) # compute costs
# activation function
optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost) # construct an optimizer
# outcome
predict_op = tf.argmax(predict_data, 1)

normal_results = []
train_normal_results = []
size = int(len(data)*0.8)
trX, teX = data[:size], data[size:]
trY, teY = labels[:size], labels[size:]

with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    batch_size = 128
    odd = []
    for i in range(0,20):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
            sess.run(optimizer, feed_dict={X: trX[start:end], Y: trY[start:end]})
        train_accuracy = np.mean(np.argmax(trY, axis = 1) == sess.run(predict_op, feed_dict={X: trX}))
        print("The train accuracy",train_accuracy)
        train_normal_results.append(train_accuracy)
        outcome = sess.run(predict_op, feed_dict={X: teX})
        accuracy = np.mean(np.argmax(teY, axis = 1) == outcome)
        print(i,accuracy)
        normal_results.append(accuracy)



print("WORKING ON SPEICAL MODEL")
predict_data = special_model(J,0.15,0.5)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict_data, labels = Y)) # compute costs
optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost) # construct an optimizer
predict_op = tf.argmax(predict_data, 1)

trX, teX = special_data[:size], special_data[size:]

special_results = []
train_special_results = []
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    batch_size = 128
    for i in range(0,20):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
            sess.run(optimizer, feed_dict={J: trX[start:end], Y: trY[start:end]})
        train_accuracy = np.mean(np.argmax(trY, axis = 1) == sess.run(predict_op, feed_dict={J: trX}))
        print("The train accuracy",train_accuracy)
        train_special_results.append(train_accuracy)
        outcome = sess.run(predict_op, feed_dict={J: teX})
        accuracy = np.mean(np.argmax(teY, axis = 1) == outcome)
        print(i,accuracy)
        special_results.append(accuracy)

#drawing graphs
list = [i for i in range(1,21)]
plt.plot(list,normal_results ,color = "green",label = "test")
plt.plot(list,train_normal_results,color = "red",label = "train")
plt.legend()
plt.title("The result of the first strategy (TOKENIZER)")
plt.xlabel('epoch', fontsize=10)
plt.ylabel('accuracy', fontsize=10)
plt.show()


plt.plot(list,special_results ,color = "green",label = "test")
plt.plot(list,train_special_results,color = "red",label = "train")
plt.legend()
plt.title("The result of the adjective words from WorldNet")
plt.xlabel('epoch', fontsize=10)
plt.ylabel('accuracy', fontsize=10)
plt.show()

print(special_results)
