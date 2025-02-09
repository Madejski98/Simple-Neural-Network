import numpy as np

class Simple_Neural_Network():

    def __init__(self, layer_dims = 784, learning_rate = 0.01):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate

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


    def train(self):
        pass

    def predict(self):
        pass

class Layer():

    def __init__(self,x,active_func, input=784, output=128):
        self.input = input
        self.output = output
        self.x = x
        self.active_func = active_func

        self.initialize_parameters()

    def initialize_parameters(self):

        he = np.sqrt(2/self.input)
        self.w = np.random.randn(self.input, self.output) * he
        self.b = np.zeros(self.output)

    def forward_propagation(self):
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
        for i in range(self.new_x.shape[0]):
            sample_sum = 0
            for j in range(self.new_x.shape[1]):
                sample_sum = sample_sum + y_train[i,j] * np.log(self.new_x[i,j] + 0.0000000001)
            sample_sum = sample_sum * (-1)
            self.loss = self.loss + sample_sum
        self.loss = self.loss/self.new_x.shape[0]
        print(f"Koszt: {self.loss}")







S=Simple_Neural_Network()
S.train_data_download()
F_L = Layer(S.x_train, 'relu')
F_L.forward_propagation()
S_L = Layer(F_L.new_x,'relu', F_L.output, 64)
S_L.forward_propagation()
O_L = Layer(S_L.new_x,'softmax', S_L.output, 10)
O_L.forward_propagation()
O_L.Categorical_Cross_Entropy(S.y_train_one_hot)