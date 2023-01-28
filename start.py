import numpy as np
import matplotlib.pyplot as plt

raw_data = np.load('data2d.npz')
X1 = raw_data['X']
y1 = raw_data['y']
print(X1.shape)
print(y1.shape)
print(X1)
print(y1)
plt.plot(X1[:,0], y1 ,'o', color='b', markersize=3)
plt.plot(X1[:,1], y1, 'o', color='r', markersize=3)
plt.show()

raw_data = np.load('data5d.npz')
X = raw_data['X']
y = raw_data['y']
print(X)
print(y)
plt.plot(X[:, 1], y, 'o', color='b', markersize=3)
plt.plot(X[:, 2], y, 'o', color='r', markersize=3)
plt.show()
