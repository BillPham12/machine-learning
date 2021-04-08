import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import random
"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""
def read(dataset="training", path="./"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        struct.unpack(">II", flbl.read(8))
        nums = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        imgs = np.fromfile(fimg, dtype=np.uint8).reshape(len(nums), rows, cols)

    # num is number
    # img is the image of the number above img[1] = lbl[1], ..., img[n] = lbl[n]
    output = []
    for i in range(10):
        zipped_file = zip(nums, imgs)
        output.append((i, list(map(lambda x: np.ravel(x[1]), filter(lambda x: x[0] == i, zipped_file)))))
    return output


train = read('training')
test = read('testing')
A = []
test_list = []
# A = {[0],[1],[2], ..., [10]}
# randomly put #567 images for each number in a column
for x in range(10):
    current = np.zeros((28 * 28, 0))
    for y in range(567):
        index = random.randrange(len(train[x][1]))
        current = np.c_[current,train[x][1][index]]
    A.append(current)
# test cases
tests = []
# getting 123 test cases for the digit <sample>
sample = 5;
for x in range(123):
    index = random.randrange(len(test[sample][1]))
    tests.append((sample, test[sample][1][index]))

base = [1,3,4,7,9,10] + list(range(12,50,3))
table = []
# I matrix 784x 784
special_matrix = np.identity(784)

# table has form [basis's number][UkUkT's values]
for k in base:
    table.append((k, []))
    print("calculating base ", k)
    element = []
    for aj in A:
        U, s, VT = np.linalg.svd(aj)
        # calculating Uk UkT, and add to the current base.
        table[len(table) - 1][1].append(np.dot(U[:,:k],U[:,:k].T))

outcome= []
outcome.append([])
outcome.append([])
for x in table:
    num_corrects = 0
    cases = 0
    for test in tests:
        array = []
        index = 0
        for base_element in x[1]:
            array.append((index,np.linalg.norm(np.dot(special_matrix - base_element,test[1]))))
            index = index +1
        # sort the array to find the smallest (I - UkUkT)z, z is test[1]
        array.sort(key=lambda element: element[1])
        if array[0][0] == test[0]:
            num_corrects = num_corrects + 1
        cases = cases + 1
    print("For base: ", x[0], " Correct: ", num_corrects, " Total test cases: ", cases, " Percentage: ", (num_corrects/cases))
    outcome[0].append(x[0])
    outcome[1].append((num_corrects/cases) * 100)
# draw the result
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
tick_spacing = 1
fig, ax = plt.subplots(1,1)
plt.plot(outcome[0],outcome[1])
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plt.ylabel('testing')
plt.axis([0, 50, 60, 100])
plt.show()
