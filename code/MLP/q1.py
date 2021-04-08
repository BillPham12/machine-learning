from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import math
import random
import time
np.random.seed(1)
def init_weights(shape,x):
    return tf.Variable(tf.random_normal(shape, stddev=x))

def f(x,y):
    return (np.cos(x+6*0.35*y) + 2*0.35*(x*y))

def generate_data(ran):
    outcome = []
    list_x = []
    list_y = []
    x_value = -1
    for x in range(0,ran):
        y_value = -1
        for y in range(0,ran):
            current = []
            current.append(x_value)
            current.append(y_value)
            list_x.append(x_value)
            list_y.append(y_value)
            outcome.append(current)
            y_value = y_value + 0.222222
        x_value = x_value + 0.222222
    return list_x, list_y,np.array(outcome)

def generate_data_randomly(ran):
    outcome = []
    list_x = np.linspace(-1,1,10)
    list_y =np.linspace(-1,1,10)
    np.random.shuffle(list_x)
    np.random.shuffle(list_y)
    outcome = np.vstack((list_x.flatten(), list_y.flatten())).T
    return outcome

# record the results of the function from the given data
def get_results(list):
    outcome = []
    for data in list:
        current = []
        current.append(f(data[0],data[1]))
        outcome.append(current)
    return outcome

# train data generation
list_x, list_y, train_x = generate_data(10)
np.random.shuffle(train_x)
train_y = get_results(train_x)

# test data generation
list_test_x, list_test_y, test_x = generate_data(9)
np.random.shuffle(test_x)
test_y = get_results(test_x)

num_of_layers = tf.constant(2,dtype = tf.int32)
# 2 place holders
X = tf.placeholder(shape =[None,2], dtype = tf.float32)
Y = tf.placeholder(shape = [None,1],dtype = tf.float32)

# network architecture
def model(X,layer,standard):
    w_h1 = init_weights([2, layer],standard) #  variables
    b1 = init_weights([layer],standard) #  variables
    w_h2 = init_weights([layer, 1],standard) # variables
    first_layer = tf.nn.tanh(tf.matmul(X,w_h1)+b1)
    return tf.matmul(first_layer,w_h2)

def min(x,y,z):
    orginal = [x,y,z]
    i = 0
    index = 0
    smallest = 0
    for x in orginal:
        if smallest == 0:
            smallest = x[99]
            index = 0
        elif smallest > x[99]:
            index = i
            smallest = x[99]
        i = i +1
    return index,orginal[index]
def min_list(lists):
    index = 0
    i  =0
    smallest = 99
    for element in lists:
        if smallest > element:
            smallest = element
            index = i
        i = i +1
    return index,smallest

part = "c"
data = []
batch_size = 25
graphs = []
table = []
neurons = [2,8,50]
if part == "a":
    for neu in neurons:
        #create predict data
        predict_data = model(X,neu,0.7)
        cost = tf.reduce_mean(tf.square(Y- predict_data)) # compute costs
        optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
        table.append([])
        mse_data = []
        epoch_data = []
        print("calculating MSE for ", neu," neurons")
        check = 0
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(0,1999):
                for x,y in zip(range (0,len(train_x),batch_size),range(batch_size, len(train_y)+1,batch_size)):
                    sess.run([cost,optimizer], feed_dict={X: train_x[x:y], Y: train_y[x:y]})
                if sess.run(cost, feed_dict = {X:train_x, Y: train_y}) < 0.02:
                    if check < 3:
                        print("Converged at epoch", i, "with MSE", sess.run(cost, feed_dict = {X:train_x, Y: train_y}))
                        mse_data.append(sess.run(cost, feed_dict = {X:test_x,Y:test_y}))
                        epoch_data.append(i)
                    check = check +1
                if i % 300 == 0:
                    print("The current MSE from the train data ",sess.run(cost, feed_dict = {X:test_x, Y: test_y}))


            table[len(table)-1].append(mse_data)
            table[len(table)-1].append(epoch_data)
            print("\n The table is defined as")
            print(table[len(table)-1],"\n")
            u = np.linspace(-1, 1, 9)
            x_ax,y_ax = np.meshgrid(u,u)

            graph = np.vstack((x_ax.flatten(), y_ax.flatten())).T
            K = f(x_ax,y_ax)
            data = sess.run(predict_data, feed_dict={X: graph})
            Z = np.reshape(data,(-1,9))
            graphs.append(Z)


    u = np.linspace(-1, 1, 9)
    x_ax,y_ax = np.meshgrid(u,u)
    ax = plt.subplot()
    # black one
    CS1 = ax.contour(x_ax,y_ax,K, colors = 'black')
    i  = 0
    for value in graphs:
        # green one
        if i == 0:
            CS2 = ax.contour(x_ax,y_ax,value, colors = 'green')
        elif i == 1:
            CS3 = ax.contour(x_ax,y_ax,value, colors = 'red')
        else:
            CS4 = ax.contour(x_ax,y_ax,value, colors = 'blue')
        i = i +1
    # table  = [mse,epochs]
    #for data in table:
        #plt.plot(data[0],data[1])
    plt.show()
elif part == "b":
    strategies = ["Gradient", "Momentum","RMS"]
    important = []
    table =[] # strategy -> neurons -> time and MSE
    for strategy in strategies:
        print("USING",strategy,"\n")
        current = []
        first_data = []
        first = 0
        for neu in [8]:
            #create predict data
            predict_data = model(X,neu,0.8)
            MSE = 0.02
            range_epochs= 100
            cost = tf.reduce_mean(tf.square(Y- predict_data)) # compute costs
            if strategy == "Gradient":
                optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(cost) # construct an optimizer
            elif strategy == "Momentum":
                optimizer = tf.train.MomentumOptimizer(0.02,0.02).minimize(cost) # construct an optimizer
            else:
                optimizer = tf.train.RMSPropOptimizer(0.02).minimize(cost)
            current.append([])
            print("calculating MSE for strategy", strategy)
            mse_data = []
            timer = []
            batch_size = 5
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                for i in range(0,range_epochs):
                    start = time.time()
                    for x,y in zip(range (0,len(train_x),batch_size),range(batch_size, len(train_y)+1,batch_size)):
                        sess.run([cost,optimizer], feed_dict={X: train_x[x:y], Y: train_y[x:y]})
                    mse_data.append(sess.run(cost, feed_dict = {X:test_x,Y:test_y}))
                    if sess.run(cost, feed_dict = {X:train_x, Y: train_y}) < MSE:
                        if first < 3:
                            current_value = sess.run(cost, feed_dict = {X:test_x, Y: test_y})
                            if first == 0:
                                important.append(current_value)
                            current_value = sess.run(cost, feed_dict = {X:test_x, Y: test_y})
                            print("The convergence is reached"
                             ,"at epoch", i, "with MSE", current_value )
                            first = first + 1
                            first_data.append((current_value,i))
                            print("Converged at epoch", i, "with MSE", sess.run(cost, feed_dict = {X:test_x, Y: test_y}))
                    stop = time.time()
                    if i % 20 == 0:
                        print("The current MSE from the train data ",sess.run(cost, feed_dict = {X:test_x, Y: test_y}))
                    timer.append(stop-start)

                current[(len(current))-1].append(mse_data)
                current[(len(current))-1].append(timer)
                print(first_data)
        table.append(current)

    print("\n The table is defined as")


    # gradient
    list = [i for i in range(1,101)]
    plt.figure(1)
    fig1 = plt.plot(list,table[0][0][0],color = "green",label = "gradient") # gradient
    fig1 = plt.plot(list,table[1][0][0],color = "red",label = "momentum") # momentum_index
    fig1 = plt.plot(list,table[2][0][0],color = "blue", label = "RMS") # rms
    plt.legend()
    plt.title("COMPARISION BETWEEN: Gradient; Momentum, RMS")
    plt.xlabel('epoch', fontsize=10)
    plt.ylabel('MSE', fontsize=10)
    smallest_index, smallest_one =min(table[0][0][0],table[1][0][0],table[2][0][0])
    print("The smallest MSE at epoch 100 is ", smallest_one[99],"with the strategy is",strategies[smallest_index])
    smallest_index, smallest_one =min_list(important)
    print("The smallest MSE when the training error is reached is", smallest_one,"with the strategy is",strategies[smallest_index])

    barWidth = 1

    plt.figure(2)
    list = [i for i in range(0,300,3)]
    list1 = [i for i in range(1,300,3)]
    list2 = [i for i in range(2,300,3)]
    list4 = list1+ list2 + list

    plt.bar(list,table[0][0][1],width = barWidth,color = "green",label = 'Gradient')
    plt.bar(list1,table[1][0][1],width = barWidth,color = "red",label = 'Momentum')
    plt.bar(list2,table[2][0][1],width = barWidth,color = "blue",label = 'RMS')
    plt.legend()
    plt.title("TIMING OF Gradient & MomenTum & RMS")
    plt.xlabel('epoch', fontsize=10)
    plt.ylabel('timing per epochs', fontsize=10)


    plt.show()
elif part == "c":
    for neu in [2,8,20,40,50]:
        #create predict data
        predict_data = model(X,neu,0.8)
        MSE = 0.02
        range_epochs= 100
        cost = tf.reduce_mean(tf.square(Y- predict_data)) # compute costs
        optimizer = tf.train.RMSPropOptimizer(0.02).minimize(cost)
        mse_data = []
        batch_size = 10
        first = 0
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(0,range_epochs):
                for x,y in zip(range (0,len(train_x),batch_size),range(batch_size, len(train_y)+1,batch_size)):
                    sess.run([cost,optimizer], feed_dict={X: train_x[x:y], Y: train_y[x:y]})
                mse_data.append(sess.run(cost, feed_dict = {X:test_x,Y:test_y}))
                if sess.run(cost, feed_dict = {X:train_x, Y: train_y}) < MSE and first == 0:
                    print("The",neu,"reach the convergence at epoch"
                     , i, "with MSE", mse_data[len(mse_data)-1])
                    first= 1
            table.append(mse_data)
    plt.figure(1)
    fig1 = plt.plot([i for i in range(1,len(table[0])+1)],table[0],color = "green",label = '2 neurons') # 2
    fig1=  plt.plot([i for i in range(1,len(table[1])+1)],table[1],color = "red",label = '8 neurons') # 8
    fig1 = plt.plot([i for i in range(1,len(table[2])+1)],table[2],color = "purple",label = '20 neurons') # 20
    fig1 = plt.plot([i for i in range(1,len(table[3])+1)],table[3],color = "pink",label = '40 neurons') # 40
    fig1 = plt.plot([i for i in range(1,len(table[4])+1)],table[4],color = "blue",label = '50 neurons') # 50
    plt.legend()
    plt.xlabel('epoch', fontsize=10)
    plt.ylabel('MSE', fontsize=10)


    table = []
    first = True
    invalid = 0
    validation_data = generate_data_randomly(10)
    validation_results = get_results(validation_data)
    valid_data  = []
    previous_value = 0
    attempt  =0
    special = []
    for neu in [8,8]:
        #create predict data
        predict_data = model(X,neu,1)
        cost = tf.reduce_mean(tf.square(Y- predict_data)) # compute costs
        optimizer = tf.train.RMSPropOptimizer(0.005).minimize(cost) # construct an optimizer
        table.append([])
        mse_data = []
        special_index = 0
        batch_size = 10
        print("calculating MSE for ", neu," neurons")
        if first == True:
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                for i in range(0,999):
                    for x,y in zip(range (0,len(train_x),batch_size),range(batch_size, len(train_y)+1,batch_size)):
                        sess.run([cost,optimizer], feed_dict={X: train_x[x:y], Y: train_y[x:y]})
                    mse_data.append(sess.run(cost, feed_dict = {X:test_x,Y:test_y}))
                    if sess.run(cost, feed_dict = {X:train_x, Y: train_y}) < 0.02:
                        current = sess.run(cost, feed_dict = {X:validation_data, Y: validation_results})
                        if special_index < 3:
                            special.append((0.025,sess.run(cost, feed_dict = {X:test_x, Y: test_y}),
                                current,sess.run(cost, feed_dict = {X:train_x, Y: train_y})))
                            special_index  = special_index +1
                            if special_index == 1:
                                special.append((0.025,sess.run(cost, feed_dict = {X:test_x, Y: test_y}),
                                    current,sess.run(cost, feed_dict = {X:train_x, Y: train_y})))
                                special_index  = special_index +1

                        if previous_value == 0:
                            previous_value = current
                        elif current - previous_value > 0.0025 and sess.run(cost, feed_dict = {X:test_x, Y: test_y}) < 0.02:
                            attempt  = attempt + 1
                            previous_value = current
                            print("Commit a validation error at", i," the current validation error is:", attempt)
                        else:
                            previous_value = current

                        if attempt == 10:
                            print ("break at epochs", i)
                            print(sess.run(cost, feed_dict = {X:train_x, Y: train_y}))
                            print(sess.run(cost, feed_dict = {X:validation_data, Y: validation_results}))
                            print(sess.run(cost, feed_dict = {X:test_x, Y: test_y}))
                            first = False
                            break
                first = False
                table[len(table)-1].append(mse_data)
                u = np.linspace(-1, 1, 9)
                x_ax,y_ax = np.meshgrid(u,u)

                graph = np.vstack((x_ax.flatten(), y_ax.flatten())).T
                K = f(x_ax,y_ax)
                data = sess.run(predict_data, feed_dict={X: graph})
                Z = np.reshape(data,(-1,9))
                graphs.append(Z)
        else:
            print("Calculating no early stop!")
            activate = False
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                for i in range(0,999):
                    for x,y in zip(range (0,len(train_x),batch_size),range(batch_size, len(train_y)+1,batch_size)):
                        sess.run([cost,optimizer], feed_dict={X: train_x[x:y], Y: train_y[x:y]})
                    current = sess.run(cost, feed_dict = {X:test_x,Y:test_y})
                    mse_data.append(current)
                table[len(table)-1].append(mse_data)
                u = np.linspace(-1, 1, 9)
                x_ax,y_ax = np.meshgrid(u,u)

                graph = np.vstack((x_ax.flatten(), y_ax.flatten())).T
                K = f(x_ax,y_ax)
                data = sess.run(predict_data, feed_dict={X: graph})
                Z = np.reshape(data,(-1,9))
                graphs.append(Z)



    u = np.linspace(-1, 1, 9)
    x_ax,y_ax = np.meshgrid(u,u)
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    for x in special:
        l1.append(x[0])
        l2.append(x[1])
        l3.append(x[2])
        l4.append(x[3])
    list = [0,1,2]
    plt.figure(2)
    fig2 = plt.plot(list,l1,color = "black",label = "goal")
    fig2 = plt.plot(list,l2,color = "red",label = 'test')
    fig2 = plt.plot(list,l3,color = "green",label = 'validation')
    fig2 = plt.plot(list,l4,color = "blue",label = 'training')
    plt.legend()
    plt.xlabel('2 Epochs', fontsize=10)
    plt.ylabel('MSE', fontsize=10)

    plt.figure(3)
    u = np.linspace(-1, 1, 9)
    x_ax,y_ax = np.meshgrid(u,u)
    fig3 = plt.contour(x_ax,y_ax,K, colors = 'black')
    i  = 0
    for value in graphs:
        # green one
        if i == 0:
            fig3 = plt.contour(x_ax,y_ax,value, colors = 'green')
        else:
            fig3 = plt.contour(x_ax,y_ax,value, colors = 'red')
        i = i +1
    plt.show()
