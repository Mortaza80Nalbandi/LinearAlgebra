import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

def f(x):
    return tf.math.exp((-1)*x)
def sigmoid(x):
    return 1/(1+tf.math.exp((-1)*x))
def Phi(X,w,b):
    z =sigmoid(X.reshape(1,2) @ w + b)
    return z

def C(w,b, data):
    c= 0
    for i in range(data[0].shape[0]):
        c+=2**((Phi(data[0][i],w,b) - data[1][i]))
    return c

def it_plot(w,b):
    xs = np.linspace(-5, 5, 100)
    ys = (-b - xs * w[0]) / w[1]
    plt.plot(xs, ys, color='black')
    error_no = 0
    plt.axis('square')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    for i in range(data[0].shape[0]):
        if (data[1][i] == 0):
            if (Phi(data[0][i], w, b) < 0.5):
                plt.plot(data[0][i][0], data[0][i][1], 'o', color='b', markersize=3)
            else:
                plt.plot(data[0][i][0], data[0][i][1], 'o', color='b', markersize=1)
                error_no = error_no+1
        elif (data[1][i] == 1):
            if (Phi(data[0][i], w, b) >= 0.5):
                plt.plot(data[0][i][0], data[0][i][1], 'o', color='r', markersize=3)
            else:
                plt.plot(data[0][i][0], data[0][i][1], 'o', color='r', markersize=1)
                error_no = error_no + 1
    print("error rate:",error_no/data[0].shape[0])
    plt.show()
def grad_decent(data):
    w = tf.Variable(np.array([1.0,1.0]).reshape((2,1)),shape=(2,1),dtype=float)
    b = tf.Variable([1.0])
    lambdaValue = 0.01 # this is the learning rate

    for i in range(5000):
        it_plot(w, b)
        with tf.GradientTape() as tape:
            y = C(w, b, data)
        [dl_dw, dl_db] = tape.gradient(y, [w, b])
        print(dl_dw, dl_db)
        print(w,b)
        w.assign_sub(lambdaValue * dl_dw)
        b.assign_sub(lambdaValue * dl_db)
        #print(C(w, b, data))
        #print(w,b)
        #print(dC_dw ,dC_dw_n )
        #print(dC_db ,dC_db_n)
        #print(np.linalg.norm(dC_dw-dC_dw_n))
        #print(np.linalg.norm(dC_dw-dC_dw_n)/np.linalg.norm(dC_dw))


    return b,w


def grad_decent_anim(data):
    w = tf.Variable(np.array([0.0,0.0]).reshape((2, 1)), shape=(2, 1), dtype=float)
    b = tf.Variable([2.0])
    lambdaValue = 0.01 # this is the learning rate

    for i in range(5000):
        error_no = 0
        plt.cla()
        with tf.GradientTape(persistent=True) as tape:
            y = C(w, b, data)
        [dl_dw, dl_db] = tape.gradient(y, [w, b])
        w.assign_sub(lambdaValue * dl_dw)
        b.assign_sub(lambdaValue * dl_db)
        print(dl_dw, dl_db)
        print(w, b)
        xs = np.linspace(-5, 5, 100)
        ys = (-b - xs * w[0]) / w[1]
        plt.plot(xs, ys, color='black')
        error_no = 0
        plt.axis('square')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        for i in range(data[0].shape[0]):
            if (data[1][i] == 0):
                if (Phi(data[0][i], w, b) < 0.5):
                    plt.plot(data[0][i][0], data[0][i][1], 'o', color='b', markersize=3)
                else:
                    plt.plot(data[0][i][0], data[0][i][1], 'o', color='b', markersize=1)
                    error_no = error_no + 1
            elif (data[1][i] == 1):
                if (Phi(data[0][i], w, b) >= 0.5):
                    plt.plot(data[0][i][0], data[0][i][1], 'o', color='r', markersize=3)
                else:
                    plt.plot(data[0][i][0], data[0][i][1], 'o', color='r', markersize=1)
                    error_no = error_no + 1
        plt.draw()
        plt.pause(.00001)
        if(error_no==0):
            return w,b



def plot2d():
    raw_data = tf.load('data2d.npz')
    X1 = raw_data['X']
    y1 = raw_data['y']
    for i in range(X1.shape[0]):
        if (y1[i] == 0):
            plt.plot(X1[i, 0], X1[i, 1], 'o', color='b', markersize=3)
        elif (y1[i] == 1):
            plt.plot(X1[i, 0], X1[i, 1], 'o', color='r', markersize=3)
    plt.show()


def plot5d():
    raw_data = tf.load('data5d.npz')
    X = raw_data['X']
    y = raw_data['y']
    for i in range(X.shape[0]):
        if (y[i] == 0):
            plt.plot(X[i, 0], X[i, 1], 'o', color='b', markersize=3)
        elif (y[i] == 1):
            plt.plot(X[i, 0], X[i, 1], 'o', color='r', markersize=3)
    plt.show()



raw_data = np.load('data2d.npz')
X1 = raw_data['X']
y1 = raw_data['y']
data = (X1,y1)
#b,w = grad_decent(data)
#print(w,b)
#it_plot(w,b)
#grad_decent(data)
w,b =grad_decent_anim(data)
#w,b= np.array([1,2]),1
#print(data)
#print("PHI",Phi(data[0],w,b))
#print("C=",C(w,b,data))
#dC_dw = compute_dC_dw(w,b, data)
#dC_db = compute_dC_db(w,b, data)
#dC_dw_n = compute_dC_dw_numeric(w,b, data)
#dC_db_n = compute_dC_db_numeric(w,b, data)
#grad_decent(data,w,b)

