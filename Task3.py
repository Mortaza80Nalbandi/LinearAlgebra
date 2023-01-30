import numpy as np
import tensorflow as tf
import csv

def sigmoid(X):
    return 1/(1+tf.math.exp((-1)*X))

def softmax(z):
    return tf.math.exp(z) / tf.math.reduce_sum(tf.math.exp(z), axis=1, keepdims=True)
def Phi(X,Layers):
    layer1 = Layers[0]
    layer2 = Layers[1]
    layer3 = Layers[2]
    z1 =layer1.get_output(X)
    z2 = layer2.get_output(z1)
    return layer3.get_output(z2)
def C( data,Layers):
    z= tf.math.multiply((Phi(data[0],Layers) - data[1]), ((Phi(data[0],Layers) - data[1])))
    c=0
    for i in range(data[0].shape[0]):
        c += z[i]
    return c
    return c
class Layer(object):

    def get_output(self, X):
        """Perform the forward step linear transformation.
        X is the input."""
        pass
    def get_params_iter(self):
        pass

    def set_params_iter(self):
        pass



class LinearLayer(Layer):
    """The linear layer performs a linear transformation to its input."""

    def __init__(self, n_in, n_out):
        """Initialize hidden layer parameters.
        n_in is the number of input variables.
        n_out is the number of output variables."""
        self.W = tf.Variable(tf.random.normal([n_in,n_out], 0, 1, tf.float32, seed=1),shape=(n_in,n_out),dtype=float)
        self.b = tf.Variable(tf.random.normal([n_out], 0, 1, tf.float32, seed=1))

    def get_output(self, X):
        """Perform the forward step linear transformation."""
        return sigmoid((X@self.W) +self.b)

    def get_params_iter(self):
        return self.W ,self.b

    def set_params_iter(self,w_in,b_in):
        self.W = w_in
        self.b = b_in



class SoftmaxOutputLayer(Layer):
    """The softmax output layer computes the classification
    propabilities at the output."""
    def __init__(self, n_in, n_out):
        """Initialize hidden layer parameters.
        n_in is the number of input variables.
        n_out is the number of output variables."""
        self.W = tf.Variable(tf.random.normal([n_in, n_out], 0, 1, tf.float32, seed=1), shape=(n_in, n_out), dtype=float)
        self.b = tf.Variable(tf.random.normal([n_out], 0, 1, tf.float32, seed=1))
    def get_output(self, X ):
        """Perform the forward step transformation."""
        return softmax(X@ self.W+self.b)
    def get_params_iter(self):
        return self.W ,self.b
    def set_params_iter(self,w_in,b_in):
        self.W = w_in
        self.b = b_in

def grad_dec(data,Layers):
    max_nb_of_iterations = 600  # Train for a maximum of 300 iterations
    lambdaValue = 0.1  # Gradient descent learning rate
    for iteration in range(max_nb_of_iterations):
        for layer in Layers:
            w,b = layer.get_params_iter()
            with tf.GradientTape() as tape:
                y = C(data,Layers)
            [dl_dw, dl_db] = tape.gradient(y, [w, b])
            w.assign_sub(lambdaValue * dl_dw)
            b.assign_sub(lambdaValue * dl_db)
            layer.set_params_iter(w,b)
        error_no=data[0].shape[0]
        x1,x2,x3 = 0,0,0
        z1 = Layers[0].get_output(data[0])
        z2 = Layers[1].get_output(z1)
        outputs = Layers[2].get_output(z2)
        for i in range(data[0].shape[0]):
            output = outputs[i]
            if(output[0]>output[1] and output[0]>output[2]):
                output = [1,0,0]
            elif (output[1]>output[0] and output[1]>output[2]):
                output = [0, 1, 0]
            elif (output[2]>output[0] and output[2]>output[1]):
                output = [0,0,1]
            if(data[1][i][0] ==1 and output[0] ==1 ):
                x1+=1
                error_no-=1
            if (data[1][i][1] == 1 and output[1] == 1):
                x2 += 1
                error_no -= 1
            if (data[1][i][2] == 1 and output[2] == 1):
                x3 += 1
                error_no -= 1
        print("iteration ",iteration," errors:" ,error_no)


X = np.zeros((150,4),dtype=float)
Y = np.zeros((150,3),dtype=float)
with open("iris.data", 'r') as file:
    csvreader = csv.reader(file)
    i=0;
    for row in csvreader:
        X[i]= row[0:4]
        if(row[4]=='Iris-setosa'):
            Y[i] = [1,0,0]
        elif(row[4]=='Iris-versicolor'):
            Y[i] = [0,1,0]
        elif (row[4] == 'Iris-virginica'):
            Y[i] = [0,0,1]
        i+=1
        if(i==150):
            break
X = tf.Variable(X, shape=(150, 4), dtype=float)
Y =tf.Variable(Y, shape=(150, 3), dtype=float)
data = (X,Y)
n_hidden_1 = 20 #number of neurons in 1st layer
n_hidden_2 = 20 #number of neurons in 2nd layer
n_input = 4 #4 columns
n_classes = 3 #Output classes

layers = [] # Define a list of layers
# Add first hidden layer
layers.append(LinearLayer(n_input, n_hidden_1))
# Add second hidden layer
layers.append(LinearLayer(n_hidden_1, n_hidden_2))
# Add output layer
layers.append(SoftmaxOutputLayer(n_hidden_2, n_classes))
grad_dec(data,layers)

