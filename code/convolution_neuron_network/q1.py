import tensorflow as tf
import numpy as np
import pickle
from matplotlib import pyplot as plt


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
np.random.seed(1)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


batch_size = 128
test_size = 256

def model1(X,p_keep_conv, p_keep_hidden):
    first_conv = init_weights([3, 3, 3, 6])       # 3x3x3 conv, 6 outputs
    w_fc = init_weights([16*16*6, 300])   # FC 16*16*6 inputs, 300 outputs
    w_o = init_weights([300, 10])         # FC 300 inputs, 10 outputs (labels)
    l1a = tf.nn.relu(tf.nn.conv2d(X, first_conv,                       # l1a shape=(?, 32, 32, 6)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 16, 16, 6)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2 = tf.reshape(l1, [-1, w_fc.get_shape().as_list()[0]])    # reshape to (?, 16x16x6)

    l2 = tf.nn.relu(tf.matmul(l2, w_fc))
    l2 = tf.nn.dropout(l2, p_keep_hidden)

    pyx = tf.matmul(l2, w_o)
    return pyx

def model2(X,p_keep_conv, p_keep_hidden):
    first_conv = init_weights([3, 3, 3, 6])       # 3x3x3 conv, 6 outputs
    second_conv = init_weights([3, 3, 6, 36])       # 3x3x6 conv, 36 outputs
    w_fc = init_weights([8*8*36, 300])   # FC 8*8*36 inputs, 300 outputs
    w_o = init_weights([300, 10])         # FC 300 inputs, 10 outputs (labels)
    l1a = tf.nn.relu(tf.nn.conv2d(X, first_conv,                       # l1a shape=(?, 32, 32, 6)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 16, 16, 6)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2a = tf.nn.relu(tf.nn.conv2d(l1, second_conv,             # l2a shape=(?, 16, 16, 36)
                    strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 8, 8, 36)
                    strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3 = tf.reshape(l2, [-1, w_fc.get_shape().as_list()[0]])    # reshape to (?, 8x8x36)

    l3 = tf.nn.relu(tf.matmul(l3, w_fc))
    l3 = tf.nn.dropout(l3, p_keep_hidden)

    pyx = tf.matmul(l3, w_o)
    return pyx


def model3(X,p_keep_conv, p_keep_hidden):
    first_conv = init_weights([3, 3, 3, 6])       # 3x3x3 conv, 6 outputs
    second_conv = init_weights([3, 3, 6, 36])       # 3x3x6 conv, 36 outputs
    third_conv = init_weights([3, 3, 36, 32])       # 36 conv, 32 outputs
    w_fc = init_weights([4*4*32, 300])   # FC 4*4*32 inputs, 300 outputs
    w_o = init_weights([300, 10])         # FC 300 inputs, 10 outputs (labels)
    l1a = tf.nn.relu(tf.nn.conv2d(X, first_conv,                       # l1a shape=(?, 32, 32, 6)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 16, 16, 6)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2a = tf.nn.relu(tf.nn.conv2d(l1, second_conv,             # l2a shape=(?, 16, 16, 36)
                    strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 8, 8, 36)
                    strides=[1, 2, 2, 1], padding='SAME')
    l3a = tf.nn.relu(tf.nn.conv2d(l2, third_conv,             # l3a shape=(?, 8,8, 32)
                strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 32)
                strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.reshape(l3, [-1, w_fc.get_shape().as_list()[0]])    # reshape to (?, 2x2x32)

    l4 = tf.nn.relu(tf.matmul(l4, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

def find_max(results):
    x = np.argmax(results[0])
    y = np.argmax(results[1])
    z = np.argmax(results[2])
    list = [results[0][x],results[1][y],results[2][z]]
    max = np.argmax(np.array(list))
    return (max+1), list[max]

find_best = False
list = [i for i in range(1,21)]
if find_best:
    results = [[],[],[]]
    for num in range(0,3):
        X = tf.placeholder("float", [None, 32, 32, 3]) # input has form 32x32x3
        Y = tf.placeholder("float", [None, 10]) # output has form array size of 10. (10 classes)
        p_keep_conv = tf.placeholder("float")
        p_keep_hidden = tf.placeholder("float")
        print("Calculating the accuracy of ",num+1," Convolutonal Layer")
        if  num == 0:
            py_x = model1(X,p_keep_conv, p_keep_hidden)
        elif num == 1:
            py_x = model2(X,p_keep_conv, p_keep_hidden)
        else:
            py_x = model3(X,p_keep_conv, p_keep_hidden)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
        train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
        predict_op = tf.argmax(py_x, 1)
        # Launch the graph in a session
        with tf.Session() as sess:
            # you need to initialize all variables
            tf.global_variables_initializer().run()

            for i in range(20):
                training_batch = zip(range(0, len(trX), batch_size),
                                     range(batch_size, len(trX)+1, batch_size))
                for start, end in training_batch:
                    sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                                  p_keep_conv: 0.8, p_keep_hidden: 0.5})

                test_indices = np.arange(len(teX)) # Get A Test Batch
                np.random.shuffle(test_indices)
                test_indices = test_indices[0:test_size]
                output = sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                            p_keep_conv: 1.0,
                                                             p_keep_hidden: 1.0})
                r = np.mean(np.argmax(teY[test_indices], axis=1) == output)
                results[num].append(r)

    plt.figure(1)
    fig1 = plt.plot(list,results[0],color = "green",label = "1 Convolutional Layer")
    fig1 = plt.plot(list,results[1],color = "red",label = "2 Convolutional Layers")
    fig1 = plt.plot(list,results[2],color = "blue", label = "3 Convolutional Layers")
    plt.legend()
    plt.title("COMPARISION BETWEEN:1, 2 and 3 Convolutional Layers")
    plt.xlabel('epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    winner,max = find_max(results)
    print("The higest accuracy is", max, "with the number of layers",winner)
    print("The average accuracy of", 1,"layer is ",np.mean(np.array(results[0])))
    print("The average accuracy of", 2,"layers is ",np.mean(np.array(results[1])))
    print("The average accuracy of", 3,"layers is ",np.mean(np.array(results[2])))

    plt.show()



def lenet_model(X,p_keep_conv, p_keep_hidden,features):
    first_conv = init_weights([3, 3, 3, features[0]])       # 3x3x3 conv, features[0] outputs
    second_conv = init_weights([3, 3, features[0], features[1]])       # 3x3xfeatures[0] conv, features[1] outputs
    third_conv = init_weights([3,3, features[1], features[2]])       # 3x3xfeatures[1] conv, features[2] outputs
    w_fc = init_weights([4*4*features[2], 300])   # FC1 4*4features[2] inputs, 300 outputs
    w_o = init_weights([300, 10])         # FC 300 inputs, 10 outputs (labels)
    l1a = tf.nn.relu(tf.nn.conv2d(X, first_conv,                       # l1a shape=(?, 32, 32, features[0])
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.avg_pool(l1a, ksize=[1, 2, 2, 1],         # l1 shape=(?, 16, 16, features[0])
                        strides=[1, 2, 2, 1], padding='SAME')

    l2a = tf.nn.relu(tf.nn.conv2d(l1, second_conv,             # l2a shape=(?, 16, 16, features[1])
                    strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.avg_pool(l2a, ksize=[1, 2, 2, 1],        # l2 shape=(?, 8, 8, features[1])
                    strides=[1, 2, 2, 1], padding='SAME')

    l3a = tf.nn.relu(tf.nn.conv2d(l2, third_conv,             # l3a shape=(?, 8,8, features[2])
                strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.avg_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, features[2])
                strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.reshape(l3, [-1, w_fc.get_shape().as_list()[0]])    # reshape to (?,4x4xfeatures[2])
    # first FC
    l4 = tf.nn.relu(tf.matmul(l4, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

features = [[6,16,32],[16,32,64],[32,64,128]]
results = [[],[],[]]

for num in range(0,3):
    X = tf.placeholder("float", [None, 32, 32, 3]) # input has form 32x32x3
    Y = tf.placeholder("float", [None, 10]) # output has form array size of 10. (10 classes)
    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    print("Calculating the accuracy of LeNet 5 by",num+1, "strategy")
    py_x = lenet_model(X,p_keep_conv, p_keep_hidden,features[num])
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()

        for i in range(20):
            training_batch = zip(range(0, len(trX), batch_size),
                                 range(batch_size, len(trX)+1, batch_size))
            for start, end in training_batch:
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                              p_keep_conv: 0.8, p_keep_hidden: 0.5})

            test_indices = np.arange(len(teX)) # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
            output = sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                        p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})
            r = np.mean(np.argmax(teY[test_indices], axis=1) == output)
            results[num].append(r)
            print(i, r)
plt.figure(2)
fig2 = plt.plot(list,results[0],color = "green",label = "1st strategy")
fig2 = plt.plot(list,results[1],color = "red",label = "2nd strategy")
fig2 = plt.plot(list,results[2],color = "blue", label = "3rd strategy")
plt.legend()
plt.title("COMPARISION BETWEEN three strategies")
plt.xlabel('epoch', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
winner,max = find_max(results)
print("The higest accuracy is", max, "with the number of layers",winner)
print("The average accuracy of", 1,"layer is ",np.mean(np.array(results[0])))
print("The average accuracy of", 2,"layers is ",np.mean(np.array(results[1])))
print("The average accuracy of", 3,"layers is ",np.mean(np.array(results[2])))
plt.show()
