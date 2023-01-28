import numpy as np
import matplotlib.pyplot as plt

def Phi(X):
    pass

def C(w,b, data):
    pass
def compute_dC_dw(w,b, data):
    pass
def compute_dC_db(w,b, data):
    pass
def compute_dC_dw_numeric(w,b, data):
    pass
def compute_dC_db_numeric(w,b, data):
    pass
raw_data = np.load('data2d.npz')
X1 = raw_data['X']
y1 = raw_data['y']
print(X1.shape)
print(y1.shape)
print(X1)
print(y1)
for i in range(X1.shape[0]):
    if(y1[i]==0):
        plt.plot(X1[i,0], X1[i,1] ,'o', color='b', markersize=3)
    elif(y1[i]==1):
        plt.plot(X1[i,0], X1[i,1], 'o', color='r', markersize=3)
plt.show()

raw_data = np.load('data5d.npz')
X = raw_data['X']
y = raw_data['y']
print(X)
print(y)
for i in range(X.shape[0]):
    if(y[i]==0):
        plt.plot(X[i,0], X[i,1] ,'o', color='b', markersize=3)
    elif(y[i]==1):
        plt.plot(X[i,0], X[i,1], 'o', color='r', markersize=3)
plt.show()

