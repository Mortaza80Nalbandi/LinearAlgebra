import math

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))
def Phi(X,w,b):
    z =sigmoid(w.T@X + b)
    return z

def C(w,b, data):
    c= 0;
    for i in range(data[0].shape[0]):
        c+=2**((Phi(data[0][i],w,b) - data[1][i]))
    return c
def compute_dC_dw(w,b, data):
    g = 0;
    for i in range(data[0].shape[0]):
        g += -1*2 * (data[0][i]*sigmoid(w.T@data[0][i]+ b))*((Phi(data[0][i], w, b) - data[1][i]))
    return g
def compute_dC_db(w,b, data):
    g = 0;
    for i in range(data[0].shape[0]):
        g +=-1*2 * (sigmoid(w.T @ data[0][i] + b) * ((Phi(data[0][i], w, b) - data[1][i])))
    return g
def compute_dC_dw_numeric(w,b, data):
    return 0
def compute_dC_db_numeric(w,b, data):
    return 0
def grad_decent(data):
    w = np.array([1,1])
    b = 1
    lambdaValue = 0.00001  # this is the learning rate
    for i in range(10000):
        dC_dw = compute_dC_dw(w, b, data)
        dC_db = compute_dC_db(w, b, data)
        w = w + lambdaValue * dC_dw
        b = b + lambdaValue * dC_db
        #print(C(w, b, data))
        if(i%100 ==0):
            xs = np.linspace(-5, 5, 100)
            ys = (-b - xs*w[0])/w[1]
            plt.plot(xs, ys, '-r',color='black')
            for i in range(data[0].shape[0]):
                if (data[1][i] == 0):
                    plt.plot(X1[i, 0], X1[i, 1], 'o', color='g', markersize=3)
                    if (Phi(data[0][i], w, b) < 0.5):
                        plt.plot(X1[i, 0], Phi(data[0][i], w, b), 'o', color='b', markersize=3)
                    else:
                        plt.plot(X1[i, 0], Phi(data[0][i], w, b), 'o', color='b', markersize=1)
                elif (data[1][i] == 1):
                    plt.plot(X1[i, 0], X1[i, 1], 'o', color='y', markersize=3)
                    if (Phi(data[0][i], w, b) >= 0.5):
                        plt.plot(X1[i, 0], Phi(data[0][i], w, b), 'o', color='r', markersize=3)
                    else:
                        plt.plot(X1[i, 0], Phi(data[0][i], w, b), 'o', color='r', markersize=1)
            plt.show()



def plot2d():
    raw_data = np.load('data2d.npz')
    X1 = raw_data['X']
    y1 = raw_data['y']
    for i in range(X1.shape[0]):
        if (y1[i] == 0):
            plt.plot(X1[i, 0], X1[i, 1], 'o', color='b', markersize=3)
        elif (y1[i] == 1):
            plt.plot(X1[i, 0], X1[i, 1], 'o', color='r', markersize=3)
    plt.show()


def plot5d():
    raw_data = np.load('data5d.npz')
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
print(grad_decent(data))
#w,b= np.array([1,2]),1
#print(data)
#print("PHI",Phi(data[0],w,b))
#print("C=",C(w,b,data))
#dC_dw = compute_dC_dw(w,b, data)
#dC_db = compute_dC_db(w,b, data)
#dC_dw_n = compute_dC_dw_numeric(w,b, data)
#dC_db_n = compute_dC_db_numeric(w,b, data)
#grad_decent(data,w,b)
