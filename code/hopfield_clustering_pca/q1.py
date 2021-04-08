import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

x, y = fetch_openml('mnist_784', data_home = "~/MNIST_DATA", return_X_y=True)
#x = x/255
sample_size = 55000
trX, teX = x[:sample_size], x[sample_size:]
trY, teY = y[:sample_size], y[sample_size:]

# training data:
print("Re-organized training data!")
one = []
five = []
for i in range(0,len(trX)):
    if int(trY[i]) == 1:
        trX[i].shape = (1,784)
        trX[i] = np.array(trX[i])
        one.append(trX[i])
    elif int(trY[i]) == 5:
        trX[i].shape = (1,784)
        trX[i] = np.array(trX[i])
        five.append(trX[i])

one_data = []
five_data = []
for x in one:
    one_data.append(np.sign(x))
for x in five:
    five_data.append(np.sign(x))

data = five_data + one_data


train_data = []
i = 0
for vector in data:
    x = []
    for value in vector:
        if value == 0:
            value = -1
        else:
            value = 1
        x.append(value)
    if i < len(five_data):
        train_data.append((5,x))
    else:
        train_data.append((1,x))
    i+= 1
data = []
correct_label = []
for x in train_data:
    data.append(x[1])
    correct_label.append(x[0])

data = np.array(data)
size = len(five_data)
trX = data.copy()

# testing data
print("Re-organized testing data!")
one = []
five = []
for i in range(0,len(teX)):
    if int(teY[i]) == 1:
        teX[i].shape = (1,784)
        teX[i] = np.array(teX[i])
        one.append(teX[i])
    elif int(teY[i]) == 5:
        teX[i].shape = (1,784)
        teX[i] = np.array(teX[i])
        five.append(teX[i])

one_data = []
five_data = []
for x in one:
    one_data.append(np.sign(x))
for x in five:
    five_data.append(np.sign(x))

new_data = five_data + one_data

test_data = []
i = 0
for vector in new_data:
    x = []
    for value in vector:
        if value == 0:
            value = -1
        else:
            value = 1
        x.append(value)
    if i < len(five_data):
        test_data.append((5,x))
    else:
        test_data.append((1,x))
    i+= 1
data = []
correct_teY = []
for x in test_data:
    data.append(x[1])
    correct_teY.append(x[0])
num = len(five_data)
sample = 1500
teX = []
teY = []
for x in range(0,sample):
    if x % 2 == 0:
        teX.append(data[x])
        teY.append(correct_teY[x])
    else:
        teX.append(data[num+x])
        teY.append(correct_teY[num+x])
teX = np.array(teX)
#random.shuffle(teX)
teY = np.array(teY)

# create class for hopfield neurons
class Network(object):
    def __init__(self):
        self.weight = np.zeros((784,784))
    def set_weights(self,W):
        self.weight = W
    # run for prediction
    def run(self, input_data):
        vector = input_data.copy()
        for k in range(0,784):
            product = np.dot(self.weight[k],vector)
            value = 0
            if product > 0:
                value= 1
            else:
                value = -1
            if value != vector[k]:
                vector[k] = value
        return vector
# train function, to create the table for the class hopefield network
def train(network, input_data):
    my_weights = np.zeros((784,784))
    n = len(input_data)
    for x in input_data:
        my_weights += np.outer(x,x.T)

    for x in range(0,784):
        my_weights[x][x] = 0
    network.set_weights(my_weights)
# displaying image of digits
def demo(predict):
    fig1, ax = plt.subplots(1, len(predict), figsize=(10, 5))
    for i in range(0,len(predict)):
        ax[i].imshow(predict[i].reshape((28, 28)))
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    fig, ax = plt.subplots(1,len(predict), figsize=(10, 5))
    for i in range(0,len(predict)):
        ax[i].imshow(teX[i].reshape((28, 28)))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()


# testing accurary
def accurary(num,predict_data):
    ran = len(predict_data)
    error = 0
    list = trX.tolist()
    if num == 2:
        for x in range(0,ran):
            index = list.index(predict_data[x].tolist())
            if int(correct_label[index]) != int(teY[x]):
                 error += 1
        return (1- error/ran)*100


    # number of digits
    for x in range(0,ran):
        if isOne(predict_data[x]):
            if int(teY[x]):
                error += 1
        elif int(teY[x]) == 1:
            error += 1
    return (1- error/ran)*100
def isOne(input):
    vector = input.copy()
    vector=  vector.reshape(28,28)
    vector = vector.T
    for x in range(0,len(vector),4):
        num1 = 0
        num2 = 0
        num3 = 0
        num4= 0
        for y in range(0,28):
            if vector[x][y] == 1:
                num1 += 1
        for y in range(0,28):
            if vector[x+1][y] == 1:
                num2 += 1
        for y in range(0,28):
            if vector[x+2][y] == 1:
                num3 += 1
        for y in range(0,28):
            if vector[x+3][y] == 1:
                num4 += 1
        if num1 + num2 + num3 + num4 >= 28*4- 47:
            return True
    return False




train_images  = [2,3,4,5,6,7,8,9,10]
#train_images = [2]
correctness = []
# Creating train data:
for num in train_images:
    train_data = []
    for i in range(0,num):
        if i % 2 == 0:
            train_data.append(trX[i])
        else:
            train_data.append(trX[size+i])
    #create hopfield class
    network = Network()
    # train data
    train(network,train_data)
    results = []
    # appending the results
    for x in range(0,100):
        results.append(network.run(teX[x]))
    # demo result for testing purpose
    #demo(results)
    correctness.append(accurary(len(train_data),results))
print(correctness)
#test data

plt.figure(1)
fig1 = plt.plot(train_images,correctness,color = "green",label = "correct percentage") # gradient
plt.legend()
plt.title("The analysis of hopfield network")
plt.xlabel('Number of trained images', fontsize=10)
plt.ylabel('Correctness', fontsize=10)
plt.show()
