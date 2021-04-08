import csv
import numpy as np
import random
from scipy.sparse.linalg import svds
import time
rating = np.zeros((943, 1682))
# read data
data_store = []
with open('C:/Users/bill/Desktop/COMP4107/Assignment1/ml-100k/u.data') as data:
    data = csv.reader(data,delimiter = '\t')
    for row in data:
        x = int(row[0])-1
        y = int(row[1])-1
        rating[x][y] = int(row[2])
        data_store.append((x,y))
k = 14
rating = np.array(rating)
average = []
import statistics
for x in rating:
    avg = 0
    N = 0
    for y in x:
        if y != 0:
            avg = avg + y
            N = N +1
    average.append(avg/N)


def MAE(predict, reality):
    total = 0
    num = 0
    for j in range(0,943):
        for k in range(0,1682):
            if reality[j][k] != 0:
                num = num +1
                total = total + abs(reality[j][k] - (predict[j][k]+average[j]))
    return total/num

def get_train_test_data(x):
    train = np.zeros((943, 1682))
    test = np.zeros((943, 1682))
    # randomly pick x% data to train and x% to test
    l = random.sample(data_store,int(x*len(data_store)))
    for index in data_store:
        x = index[0]
        y = index[1]
        # as the lecture note to calcualte the SVD of S - avg(S)
        if index in l:
            train[x][y] = rating[x][y] - average[x]
        else:
            test[x][y] = rating[x][y]
    return train, test

training_test_ratios = [0.2,0.5,0.8]
outcome = []
t = []
for ratio in training_test_ratios:
    train, test = get_train_test_data(ratio)
    print("The current training test ratio is ", ratio)
    outcome.append([])
    t.append([])
    num = 100000*(1-ratio)
    for point in range(600,1000,50):
        if point == 950:
            point = 943
        print("Calculating from basis ",point)
        # origional U, S, VT
        k = 14
        start = time.time()
        U,S,VT = np.linalg.svd(train[:point])
        # Uk, Sk,
        Uk = U[:,:k]
        Sk =np.diag(S[:k])
        Vk = VT[:k]
        # fold_in process
        for start_point in range(point,943):
            n = np.array(train[start_point])
            x = np.dot(np.dot(n,Vk.T),np.linalg.inv(Sk))
            Uk = np.vstack([Uk,x])
        # estimation data used to predict
        UkSk = np.dot(Uk,np.sqrt(Sk).T)
        SkVk = np.dot(np.sqrt(Sk),Vk)
        predict_table = np.dot(UkSk, SkVk)
        # predict_data I J , I is customer and J is product
        print("The MAE value is", MAE(predict_table,test))
        outcome[len(outcome)-1].append(MAE(predict_table,test))
        stop = time.time()
        t[len(outcome)-1].append(num/(stop-start))


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
df=pd.DataFrame({'x': [600,650,700,750,800,850,900,943], 'x=0.2': outcome[0],'x=0.5': outcome[1],'x=0.8': outcome[2]})
# multiple line plot
plt.plot( 'x', 'x=0.2', data=df, marker='p', color='purple',linewidth=2)
plt.plot( 'x', 'x=0.5', data=df, marker='s', color='pink', linewidth=2)
plt.plot( 'x', 'x=0.8', data=df, marker='^', color='red', linewidth=2)
plt.ylabel('MAE')
plt.legend()
plt.show()



df1=pd.DataFrame({'x': [600,650,700,750,800,850,900,943], 'x=0.2': t[0],'x=0.5': t[1],'x=0.8': t[2]})

# multiple line plot
plt.plot( 'x', 'x=0.2', data=df1, marker='p', color='purple',linewidth=2)
plt.plot( 'x', 'x=0.5', data=df1, marker='s', color='pink', linewidth=2)
plt.plot( 'x', 'x=0.8', data=df1, marker='^', color='red', linewidth=2)
plt.ylabel('Suggest rating/second')
plt.legend()
plt.show()
