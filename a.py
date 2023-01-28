import numpy as np
import matplotlib.pyplot as plt

def compute_numerical_derivative(func, x, method='custom'):
    y = func(x)
    if method == 'custom':
        size = len(y0)
        res = np.zeros(size, 'd') # 'd' for double
        # centered differences
        for idx in range(1, size-1):
            res[idx] = (y[idx+1] - y[idx-1]) / (x[idx+1] - x[idx-1])
        # one-sided differences
        res[0] = (y[1] - y[0]) / (x[1] - x[0])
        res[-1] = (y[size-1] - y[size-2]) / (x[size-1] - x[size-2])
    # elif method == 'numpy':
    #     res = np.gradient(y)
    return res

x = np.linspace(0, 2*np.pi, 100)
y0 = np.sin(x)
y1_true = np.cos(x) # exactly d/dx (y0)

y1_cust = compute_numerical_derivative(np.sin, x, method='custom')
# y1_nump = compute_numerical_derivative(np.sin, x, method='numpy')

plt.plot(x, y1_true, 'r', marker='o', linestyle=None, alpha=0.5)
plt.plot(x, y1_cust, 'b', marker='^', linestyle=None, alpha=0.5)
# plt.plot(x, y1_nump, 'k', marker='*', linestyle='-', alpha=0.5)
plt.show()