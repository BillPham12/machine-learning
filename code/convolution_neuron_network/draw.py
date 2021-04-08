import tensorflow as tf
import numpy as np
import pickle
from matplotlib import pyplot as plt
import math

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# load data function
def load_data(kind):
    if kind == 'test':
        data = []
        labels = []
        kind_of_data = 'cifar-10-batches-py/test_batch'
        with open(kind_of_data, mode='rb') as file:
            train = pickle.load(file, encoding='latin1')
        label = train['labels']
        # reshape picture from 3027 to 32x32x3
        features = train['data'].reshape((len(train['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        for e in features:
            data.append(e)
        for thing in label:
            l = np.zeros(10)
            l[thing] = 1
            labels.append(l)
        return np.array(data),np.array(labels)
    else:
        data = []
        labels = []
        for x in range(1,6):
            kind_of_data = 'cifar-10-batches-py/data_batch_' + str(x)
            with open(kind_of_data, mode='rb') as file:
                train = pickle.load(file, encoding='latin1')
            label = train['labels']
            # reshape picture from 3027 to 32x32x3
            features = train['data'].reshape((len(train['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
            for e in features:
                data.append(e)
            for thing in label:
                l = np.zeros(10)
                l[thing] = 1
                labels.append(l)
    return np.array(data),np.array(labels)


# get train and test data
# train has size 50000 and test 10000
trX,trY = load_data('train')
teX,teY = load_data('test')

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trX[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
plt.show()

X = tf.placeholder("float", [None, 32, 32, 3]) # input has form 32x32x3
Y = tf.placeholder("float", [None, 10]) # output has form array size of 10. (10 classes)
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")


first_conv = init_weights([3, 3, 3, 32])       # 5x5x3 conv, 6 outputs
second_conv = init_weights([3, 3, 32, 64])       # 5x5x6 conv, 36 outputs
third_conv = init_weights([3, 3, 64, 128])       # 36 conv, 32 outputs
w_fc = init_weights([4*4*128, 300])   # FC 2*2*32 inputs, 300 outputs
w_o = init_weights([300, 10])         # FC 625 inputs, 10 outputs (labels)
l1a = tf.nn.relu(tf.nn.conv2d(X, first_conv,                       # l1a shape=(?, 32, 32, 32)
                    strides=[1, 1, 1, 1], padding='SAME'))
l1 = tf.nn.avg_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 16, 16, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
l2a = tf.nn.relu(tf.nn.conv2d(l1, second_conv,             # l2a shape=(?, 16, 16, 64)
                    strides=[1, 1, 1, 1], padding='SAME'))
l2 = tf.nn.avg_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 8, 8, 64)
                    strides=[1, 2, 2, 1], padding='SAME')
l3a = tf.nn.relu(tf.nn.conv2d(l2, third_conv,             # l3a shape=(?, 8,8, 128)
                strides=[1, 1, 1, 1], padding='SAME'))
l3 = tf.nn.avg_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                strides=[1, 2, 2, 1], padding='SAME')
l3 = tf.nn.dropout(l3, p_keep_conv)

l4 = tf.reshape(l3, [-1, w_fc.get_shape().as_list()[0]])    # reshape to (?, 4x4x128)

l4 = tf.nn.relu(tf.matmul(l4, w_fc))
l4 = tf.nn.dropout(l4, p_keep_hidden)

py_x = tf.matmul(l4, w_o)

batch_size = 128
test_size = 256

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    # source code:
    #https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4
    def getActivations(layer,stimuli):
        units = sess.run(layer,feed_dict={X: np.reshape(stimuli,[1,32,32,3],order = 'F'),p_keep_conv: 0.8, p_keep_hidden: 0.5})
        plotNNFilter(units)

    def plotNNFilter(units):
        filters = units.shape[3]
        plt.figure(1, figsize=(20,20))
        n_columns = 10
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i+1)
            plt.imshow(units[0,:,:,i],cmap=plt.cm.binary)
        plt.show()

    for i in range(20):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})
    imageToUse = trX[10]
    plt.imshow(imageToUse, cmap=plt.cm.binary)
    plt.show()
    getActivations(l1a,imageToUse)
    getActivations(l2a,imageToUse)
    getActivations(l3a,imageToUse)

    imageToUse = trX[20]
    plt.imshow(imageToUse, cmap=plt.cm.binary)
    plt.show()
    getActivations(l1a,imageToUse)
    getActivations(l2a,imageToUse)
    getActivations(l3a,imageToUse)
