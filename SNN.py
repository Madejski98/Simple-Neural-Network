import numpy as np

class Simple_Neural_Network():

    def __init__(self, layer_dims = 784, learning_rate = 0.00001, layers_num = 3):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.layers_num = layers_num
        self.layers = []

    def train_data_download(self):
        mnist_data = np.loadtxt("C:\\Users\\danie\\Desktop\\MNIST_CSV\\mnist_train.csv", delimiter=",")

        self.x_train = mnist_data[:,1:]
        self.y_train = mnist_data[:,0]

        self.x_train = self.x_train / 255
        self.y_train_one_hot = np.zeros((60000, 10))

        for i in range(self.y_train_one_hot.shape[0]):
            for j in range(self.y_train_one_hot.shape[1]):
                if j == self.y_train[i]:
                    self.y_train_one_hot[i,j] = 1

    def Layers_collect(self,layer):
        self.layers.append(layer)


    def train(self):
        layers = []

    def predict(self):
        pass

class Layer():

    def __init__(self,active_func, input=784, output=128):
        self.input = input
        self.output = output

        self.active_func = active_func

        self.initialize_parameters()

    def initialize_parameters(self):

        he = np.sqrt(2/self.input)
        self.w = np.random.randn(self.input, self.output) * he
        self.b = np.zeros(self.output)

    def forward_propagation(self, x):
        self.x = x
        self.y = self.x @ self.w + self.b
        if self.active_func == 'relu':
            self.new_x = self.ReLU(self.y)
        else:
            self.new_x = self.Softmax(self.y)

    def ReLU(self, y):
        return np.maximum(0,y)

    def Softmax(self, y):
        new_x = np.zeros((y.shape))
        for i in range(new_x.shape[0]):
            sum_exp_i = np.sum(np.exp(y[i]))
            for j in range(new_x.shape[1]):
                new_x[i,j] = np.exp(y[i,j])/sum_exp_i
        return new_x

    def Categorical_Cross_Entropy(self, y_train):
        self.loss = 0
        self.y_train = y_train
        for i in range(self.new_x.shape[0]):
            sample_sum = 0
            for j in range(self.new_x.shape[1]):
                sample_sum = sample_sum + self.y_train[i,j] * np.log(self.new_x[i,j] + 0.0000000001)
            sample_sum = sample_sum * (-1)
            self.loss = self.loss + sample_sum
        self.loss = self.loss/self.new_x.shape[0]
        print(f"Koszt: {self.loss}")

    def backward_propagation(self, last_layer_flag=False, grad=None, weights = None):
        if last_layer_flag:
            self.CCE_gradient()
            self.weights_and_bias_gradient()
        else:
            self.activation_func_gradients(grad, weights)
            self.weights_and_bias_gradient()

    def CCE_gradient(self):
        self.dZ = self.new_x - self.y_train

    def activation_func_gradients(self, grad, weights):
        self.dA = grad @ weights.T

        self.dZ = np.zeros(self.dA.shape)
        for i in range(self.y.shape[0]):
            for j in range(self.y.shape[1]):
                if self.y[i,j] > 0:
                    self.dZ[i,j] = self.dA[i,j] * 1

    def weights_and_bias_gradient(self):
        self.dw = self.x.T @ self.dZ
        self.db = np.sum(self.dZ,0)

    def weigts_update(self, learning_rate):
        self.w = self.w - (learning_rate * self.dw)
        self.b = self.b - (learning_rate * self.db)








S=Simple_Neural_Network()
S.train_data_download()
F_L = Layer('relu')
S_L = Layer('relu', F_L.output, 64)
O_L = Layer('softmax', S_L.output, 10)
F_L.forward_propagation(S.x_train)
S_L.forward_propagation(F_L.new_x)
O_L.forward_propagation(S_L.new_x)
O_L.Categorical_Cross_Entropy(S.y_train_one_hot)
O_L.backward_propagation(last_layer_flag=True)
S_L.backward_propagation(grad=O_L.dZ, weights=O_L.w)
F_L.backward_propagation(grad=S_L.dZ, weights=S_L.w)
O_L.weigts_update(S.learning_rate)
S_L.weigts_update(S.learning_rate)
F_L.weigts_update(S.learning_rate)
F_L.forward_propagation(S.x_train)
S_L.forward_propagation(F_L.new_x)
O_L.forward_propagation(S_L.new_x)
O_L.Categorical_Cross_Entropy(S.y_train_one_hot)
O_L.backward_propagation(last_layer_flag=True)
S_L.backward_propagation(grad=O_L.dZ, weights=O_L.w)
F_L.backward_propagation(grad=S_L.dZ, weights=S_L.w)
O_L.weigts_update(S.learning_rate)
S_L.weigts_update(S.learning_rate)
F_L.weigts_update(S.learning_rate)
for i in range(50):
    F_L.forward_propagation(S.x_train)
    S_L.forward_propagation(F_L.new_x)
    O_L.forward_propagation(S_L.new_x)
    O_L.Categorical_Cross_Entropy(S.y_train_one_hot)
    O_L.backward_propagation(last_layer_flag=True)
    S_L.backward_propagation(grad=O_L.dZ, weights=O_L.w)
    F_L.backward_propagation(grad=S_L.dZ, weights=S_L.w)
    O_L.weigts_update(S.learning_rate)
    S_L.weigts_update(S.learning_rate)
    F_L.weigts_update(S.learning_rate)