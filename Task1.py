import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def init(self):
        pass
    def f(self,x):
        return np.exp((-1) * x)


    def sigmoid(self,x):
        return 1 / (1 + np.exp((-1) * x))


    def Phi(self,X, w, b):
        z = self.sigmoid(w.T @ X + b)
        return z


    def C(self,w, b, data):
        c = 0
        for i in range(data[0].shape[0]):
            c += (self.Phi(data[0][i], w, b) - data[1][i]) * (self.Phi(data[0][i], w, b) - data[1][i])
        return c


    def compute_dC_dw(self,w, b, data):
        g = 0
        for i in range(data[0].shape[0]):
            g += 2 * (data[0][i] * self.f(w.T @ data[0][i] + b) * (
                    self.sigmoid(w.T @ data[0][i] + b) * self.sigmoid(w.T @ data[0][i] + b))) * (
             (self.Phi(data[0][i], w, b) - data[1][i]))
        return g


    def compute_dC_db(self,w, b, data):
        g = 0
        for i in range(data[0].shape[0]):
            g += 2 * self.f(w.T @ data[0][i] + b) * (self.sigmoid(w.T @ data[0][i] + b) * self.sigmoid(w.T @ data[0][i] + b) * (
            (self.Phi(data[0][i], w, b) - data[1][i])))
        return g


    def compute_dC_dw_numeric(self,w, b, data):
        epsilon = 0.00001
        c1 = self.C(w, b, data)
        c2 = self.C(w + epsilon, b, data)
        return (c2 - c1) / epsilon


    def compute_dC_db_numeric(self,w, b, data):
        epsilon = 0.00001
        c1 = self.C(w, b, data)
        c2 = self.C(w, b + epsilon, data)
        return (c2 - c1) / epsilon


    def it_plot(self,w, b):
        xs = np.linspace(-5, 5, 100)
        ys = (-b - xs * w[0]) / w[1]
        plt.plot(xs, ys, color='black')
        error_no = 0
        plt.axis('square')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        for i in range(data[0].shape[0]):
            if data[1][i] == 0:
                if self.Phi(data[0][i], w, b) < 0.5:
                    plt.plot(data[0][i][0], data[0][i][1], 'o', color='b', markersize=3)
                else:
                    plt.plot(data[0][i][0], data[0][i][1], 'o', color='b', markersize=1)
                    error_no = error_no + 1
            elif data[1][i] == 1:
                if self.Phi(data[0][i], w, b) >= 0.5:
                    plt.plot(data[0][i][0], data[0][i][1], 'o', color='r', markersize=3)
                else:
                    plt.plot(data[0][i][0], data[0][i][1], 'o', color='r', markersize=1)
                    error_no = error_no + 1
        print("error rate:", error_no / data[0].shape[0])
        plt.show()


    def anim_plot(self,w, b, data):
        plt.cla()
        xs = np.linspace(-10, 10, 100)
        ys = (-b - xs * w[0]) / w[1]
        plt.title("Learning Representation")
        plt.plot(xs, ys, color='black', label='Classifier line')
        error_no = 0
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.plot([], [], 'o', color='r', markersize=3, label='True Positive')
        plt.plot([], [], 'o', color='r', markersize=1, label='False Negative')
        plt.plot([], [], 'o', color='b', markersize=3, label='True Negative')
        plt.plot([], [], 'o', color='b', markersize=1, label='False Positive')
        for i in range(data[0].shape[0]):
            if data[1][i] == 0:
                if self.Phi(data[0][i], w, b) < 0.5:
                    plt.plot(data[0][i][0], data[0][i][1], 'o', color='b', markersize=3)
                else:
                    plt.plot(data[0][i][0], data[0][i][1], 'o', color='b', markersize=1)
                    error_no += 1
            elif data[1][i] == 1:
                if self.Phi(data[0][i], w, b) >= 0.5:
                    plt.plot(data[0][i][0], data[0][i][1], 'o', color='r', markersize=3)
                else:
                    plt.plot(data[0][i][0], data[0][i][1], 'o', color='r', markersize=1)
                    error_no += 1
        plt.legend()
        plt.draw()
        plt.pause(0.001)
        return error_no


    def grad_decent(self,data):  # this is the implementation of algorithm with no visual animation
        w = np.array([4.5, 3.5])
        b = 2.0
        lambdaValue = 0.01  # this is the learning rate
        for i in range(5000):
            dC_dw = self.compute_dC_dw(w, b, data)
            dC_db = self.compute_dC_db(w, b, data)
            tempw = w - lambdaValue * dC_dw
            tempb = b - lambdaValue * dC_db
            error_no = 0
            for i in range(data[0].shape[0]):
                if data[1][i] == 0:
                    if self.Phi(data[0][i], w, b) >= 0.5:
                        error_no += 1
                elif data[1][i] == 1:
                    if self.Phi(data[0][i], w, b) < 0.5:
                        error_no += 1
            if abs(w[0] - tempw[0]) < 0.0000001 and abs(w[1] - tempw[1]) < 0.0000001 and abs(
                    b - tempb) < 0.0000001 and error_no == 0:
                print("break", i)
                break
            w = tempw
            b = tempb
        return b, w


    def grad_decent_anim(self,data):
        # Initialize the weights
        w = np.array([4.5, 3.5])
        b = 2.0
        lambdavalue = 0.1  # this is the learning rate
        for episode in range(5000):
            error_no = self.anim_plot(w, b, data)
            dc_dw = self.compute_dC_dw(w, b, data)
            dc_db = self.compute_dC_db(w, b, data)
            tempw = w - lambdavalue * dc_dw
            tempb = b - lambdavalue * dc_db
            if abs(w[0] - tempw[0]) < 0.001 and abs(w[1] - tempw[1]) < 0.001 and abs(b - tempb) < 0.001 and error_no == 0:
                print("Termination in Episode : ", episode)
                break
            w = tempw
            b = tempb
            print(
                f'Epoch {episode + 1}, Errors happened: {error_no}, Accuracy: {round(((data[0].shape[0] - error_no) / data[0].shape[0]) * 100, 2)}%')
        return w, b


def plotXd(filename):
    raw_data = np.load(filename + '.npz')
    plt.title(filename + " Representation")
    plt.xlabel("X0")
    plt.ylabel("X1")
    X1 = raw_data['X']
    y1 = raw_data['y']
    plt.plot([], [], 'o', color='r', markersize=3, label='Positive')
    plt.plot([], [], 'o', color='b', markersize=3, label='Negative')
    for i in range(X1.shape[0]):
        if y1[i] == 0:
            plt.plot(X1[i, 0], X1[i, 1], 'o', color='b', markersize=3)
        elif y1[i] == 1:
            plt.plot(X1[i, 0], X1[i, 1], 'o', color='r', markersize=3)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plotXd("data2d")
    plotXd("data5d")
    raw_data = np.load('data2d.npz')
    X1 = raw_data['X']
    y1 = raw_data['y']
    data = (X1, y1)
    logreg = LogisticRegression()
    w, b = logreg.grad_decent_anim(data)
    print(f"W = {w} \n b = {b}")
