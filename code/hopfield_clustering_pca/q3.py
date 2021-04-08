import numpy as np
import random
import sklearn
import sklearn.cluster as sk
from sklearn.utils import shuffle
from sklearn.datasets import fetch_openml
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



x, y = fetch_openml('mnist_784', data_home = "~/MNIST_DATA", return_X_y=True)
#x = x/255
trX, teX = x[:60000], x[60000:]
trY, teY = y[:60000], y[60000:]
one = []
five = []
for i in range(0,len(x)):
    if int(y[i]) == 1:
        x[i].shape = (1,784)
        x[i] = np.array(x[i])
        one.append(x[i])
    elif int(y[i]) == 5:
        x[i].shape = (1,784)
        x[i] = np.array(x[i])
        five.append(x[i])

one_data = []
five_data = []
for x in one:
    one_data.append(np.sign(x))
for x in five:
    five_data.append(np.sign(x))

random.seed(1)
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

# data contains 1 and 5
data = np.array(data)
correct_label = np.array(correct_label)
data,correct_label = shuffle(data, correct_label)


def draw_som(som):
    plt.figure(figsize=(5, 5))
    for i in range (0,len(data)):
        x, y = som.winner(data[i])
        plt.text(x, y, str(correct_label[i]), color= "blue")
    plt.axis([0, 30, 0, 30])
    plt.show()

def draw_kmean(som):
    plt.figure(figsize=(5, 5))
    for i in range (0,len(data)):
        x, y = som.winner(data[i])
        plt.text(x, y, str(correct_label[i]), color= "blue")
    plt.axis([0, 30, 0, 30])
    plt.show()

# the som has size of 30x30, the learning rate is 0.1 and the radius is 2
som = MiniSom(30,30,784,sigma = 2, learning_rate= 0.1)
# train data with iteration 4999
som.train(data,4999)
print("Drawing SOM after training with 4999 iterations")
draw_som(som)



# k means
k = 2

# source code:
#https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
pca_data = PCA(n_components = 2).fit_transform(data)
kmean = sk.KMeans(n_clusters=k, n_init=1).fit(pca_data)
centroids = kmean.cluster_centers_
print("Drawing kmeans")

x_min, x_max = 99999,-99999
y_min, y_max = 99999,-99999
for x,y in pca_data:
    if x_max < x:
        x_max = x
    if x_min > x:
        x_min  = x
    if y_max < y:
        y_max = y
    if y_min > y:
        y_min  = y
x_min -= 1
x_max +=1
y_min -=1
y_max += 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))


plt.figure(1)
plt.clf()
Z = kmean.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.imshow(Z,interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
i = 0
for x,y in pca_data:
    plt.text(x,y,correct_label[i],  color= "blue")
    i += 1
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='White', zorder=10)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
