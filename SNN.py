"""
Projekt Prostej Sieci Neuronowej wykonanej do rozpoznawania danych liczb
napisanych odręcznie (dane MNIST).
Sieć wykonana samodzielnie bez użycia zewnętrznych bibliotek i frameworków.
Wykonał: Daniel Madejski
Biblioteki: Numpy
"""

import numpy as np

class Simple_Neural_Network():
    """
    Klasa reprezentująca sieć neuronową.
    Zawiera metody inicjalizacji, pobierania danych, zbierania warstw, trenowania i predykowania.
    """
    def __init__(self, layer_dims = 784, learning_rate = 0.00001, epoch=50):
        """
        Metoda inicjalizacyjna.
        Inicjalizuje klasę Simple_Neural_Network.
        Przyjmuje zmienne:
            -layer_dims - wielkość warstwy wiejściowej (784 ze względu na dane MNIST)
            -leatning_rate - krok uczenia (0.00001 ponieważ model ma tendencję niestabilności przy większej wartości)
            -layers - lista przygotowana do przechowywania warstw sieci
            -epoch - liczba iteracji przez sieć
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.layers = []
        self.epoch = epoch

    def train_data_download(self):
        """
        Metoda pobierająca i normalizująca dane które w dokumencie csv już są flatennowane.
        Dzieli pobrane dane Mnist na macierz 60000 próbek o ilości pikseli 784 (28 x 28)
        oraz wektor etykiet, które następnie zmieniane są na macierz One Hot.
        Dane w macierzy x są normalizowane dzieląc przez 255 co jest maksymalną wartością.
        """
        mnist_data = np.loadtxt("mnist_train.csv", delimiter=",")

        self.x_train = mnist_data[:,1:]
        self.y_train = mnist_data[:,0]

        self.x_train = self.x_train / 255
        self.y_train_one_hot = np.zeros((60000, 10))

        for i in range(self.y_train_one_hot.shape[0]):
            for j in range(self.y_train_one_hot.shape[1]):
                if j == self.y_train[i]:
                    self.y_train_one_hot[i,j] = 1

    def layers_collect(self,layer):
        """Metoda dodająca warstwy do listy 'self.layers'."""
        self.layers.append(layer)

    def train(self):
        """
        Metoda trenująca sieć neuronową.
        Pętla wykonująca się podaną przez użytkownika ilość razy. (self.epoch)
        Posiada trzy pętle wewnętrzne
            1. Wykonująca propagację w przód.
            2. Przeprowadzająca propagację w tył.
            3. Aktualizująca wagi i biasy.
        """
        for i in range(self.epoch):
            print(f"[EPOCH {i+1}/{self.epoch}] ", end='')
            for l in range(len(self.layers)):
                if l == 0:
                    self.layers[l].forward_propagation(self.x_train)
                else:
                    self.layers[l].forward_propagation(self.layers[l-1].new_x)
            for l2 in range(len(self.layers)-1, -1, -1):
                if l2 == len(self.layers)-1:
                    self.layers[l2].Categorical_Cross_Entropy(self.y_train_one_hot)
                    self.layers[l2].backward_propagation(last_layer_flag=True)
                else:
                    self.layers[l2].backward_propagation(grad=self.layers[l2+1].dZ, weights=self.layers[l2+1].w)
            for l3 in range(len(self.layers) - 1, -1, -1):
                self.layers[l3].weights_update(self.learning_rate)


    def predict(self):
        """
        Metoda predykująca wykonywana na zbiorze testowym.
        Przygotowuje dane testowe dla sieci, a następnie
        przeprowadza propagację w przód na zbiorze testowym.
        Porównuje wyniki z etykietami i podaje poziom dokładności.
        """
        test_data = np.loadtxt("mnist_test.csv", delimiter=",")
        x_predict = test_data[:, 1:]
        y_predict = test_data[:, 0]

        y_predict_one_hot = np.zeros((y_predict.shape[0], 10))
        for i in range(y_predict_one_hot.shape[0]):
            for j in range(y_predict_one_hot.shape[1]):
                if j == y_predict[i]:
                    y_predict_one_hot[i,j] = 1

        x_predict = x_predict / 255
        for l in range(len(self.layers)):
            if l == 0:
                self.layers[l].forward_propagation(x_predict)
            else:
                self.layers[l].forward_propagation(self.layers[l - 1].new_x)

        acc = 0
        for i in range(y_predict_one_hot.shape[0]):
            if np.argmax(self.layers[-1].new_x[i]) == np.argmax(y_predict_one_hot[i]):
                acc += 1

        acc = acc/y_predict_one_hot.shape[0]
        print(f"Accuracy on test data frame = {acc}")





class Layer():
    """
    Klasa reprezentująca warstwe sieci neuronowej.
    Zawiera nasepujące metody:
        - __init__ - inicjalizacja klasy,
        - initialize_parameters - inicjalizacja wag i biasów,
        - forward_propagation - propagacja w przód,
        - ReLU - wykonanie funkcji ReLU,
        - Softmax - wykonanie funkcji Softmax,
        - Categorical_Cross_Entropy - entropia krzyżowa,
        - backward_propagation - propagacja w tył,
        - CCE_gradient - gradient funkcji kosztu,
        - activation_func_gradient - gradient funkcji aktywacji,
        - weights_and_bias_gradient - gradient wag i biasów,
        - weights_update - aktualizacja wag i biasów.
    """

    def __init__(self,active_func, input=784, output=128):
        """
        Metoda inicjalizacyjna.
        Inicjalizuje klasę Layer
        oraz uruchamia metodę initialize_parameters.
        Przyjmuje zmienne:
            - active_func - nazwa funkcji aktywacji,
            - input - wielkość wejścia warstwy
            - output - wielkość wyjścia warstwy
        """
        self.input = input
        self.output = output

        self.active_func = active_func

        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Metoda inicjalizująca wag metodą He.
        """
        he = np.sqrt(2/self.input)
        self.w = np.random.randn(self.input, self.output) * he
        self.b = np.zeros(self.output)

    def forward_propagation(self, x):
        """
        Metoda wykonująca propagację w przód.
        Przyjmuje parametr x jako wejście warstwy.
        Wylicza wyjście warstwy które podawane jest na odpowiednią
        funkcję aktywacji przed przekazaniem na następną warstwę..
        """
        self.x = x
        self.y = self.x @ self.w + self.b
        if self.active_func == 'relu':
            self.new_x = self.ReLU(self.y)
        else:
            self.new_x = self.Softmax(self.y)

    def ReLU(self, y):
        """
        Metoda wykonująca funkcję aktywacji ReLU.
        Przyjmuje zmienną y, która reprezenuje wyjście podane na
        funkcję aktywacji.
        Daje 0 dla wartości mniejszych niż 0 i a przekazuje
        wartości w niezmienionej formie które są większe od 0.
        """
        return np.maximum(0,y)

    def Softmax(self, y):
        """
        Metoda wykonująca funkcjęaktywacji Softmax.
        Przyjmuje zmienną y, która reprezenuje wyjście podane na
        funkcję aktywacji.
        Wyjście aktywowane tą funkcją daje wartości
        sumujące się do 1 co daje prawdopodobieństwo,
        że rzecz podana na wejście sieci
        odpowiada rzeczywistej konkretnej klasie.
        """
        new_x = np.zeros((y.shape))
        for i in range(new_x.shape[0]):
            sum_exp_i = np.sum(np.exp(y[i]))
            for j in range(new_x.shape[1]):
                new_x[i,j] = np.exp(y[i,j])/sum_exp_i
        return new_x

    def Categorical_Cross_Entropy(self, y_train):
        """
        Metoda wykonująca entropię krzyżową.
        Jest to funkcja kosztu aktywacji Softmax,
        rozumiana jako różnica między rozkładem prawdopodobieństwa
        czyli wyjściem sieci, a rzeczywistymi etykietami.
        """
        self.loss = 0
        self.y_train = y_train
        for i in range(self.new_x.shape[0]):
            sample_sum = 0
            for j in range(self.new_x.shape[1]):
                sample_sum = sample_sum + self.y_train[i,j] * np.log(self.new_x[i,j] + 0.0000000001)
            sample_sum = sample_sum * (-1)
            self.loss = self.loss + sample_sum
        self.loss = self.loss/self.new_x.shape[0]
        print(f"COST: {self.loss}")

    def backward_propagation(self, last_layer_flag=False, grad=None, weights = None):
        """
        Metoda wykonująca porpagację w tył.
        Przyjmuje zmienne:
            - last_layer_flag - flaga ostatniej klasy (domyślnie False),
            - grad - gradient poprzedniej warstwy (domyślnie None),
            - weights - wagi poprzedniej warstwy (domyślnie None)
        Wywołuje funkcje wyliczania gradientu funkcji kosztu dla ostatniej warstwy,
        metodę obliczającą gradient funkcji aktywacji dla pozostałych warstw
        oraz metodę wyliczającą gradient wag i biasów dla każdej z warstw.
        """
        if last_layer_flag:
            self.CCE_gradient()
            self.weights_and_bias_gradient()
        else:
            self.activation_func_gradients(grad, weights)
            self.weights_and_bias_gradient()

    def CCE_gradient(self):
        """
        Metoda licząca gradient funkcji kosztu.
        """
        self.dZ = self.new_x - self.y_train

    def activation_func_gradients(self, grad, weights):
        """
        Metoda wyliczająca gradient funkcji aktwyacji.
        Przyjmuje zmienne grad i weights jako gradient i wagi poprzedniej warstwy.
        """
        self.dA = grad @ weights.T

        self.dZ = np.zeros(self.dA.shape)
        for i in range(self.y.shape[0]):
            for j in range(self.y.shape[1]):
                if self.y[i,j] > 0:
                    self.dZ[i,j] = self.dA[i,j] * 1

    def weights_and_bias_gradient(self):
        """
        Metoda licząca gradient wag i biasów.
        """
        self.dw = self.x.T @ self.dZ
        self.db = np.sum(self.dZ,0)

    def weights_update(self, learning_rate):
        """
        Metoda aktualizująca wagi i biasy.
        Przyjmuje zmienną learning_rate (krok uczenia).
        """
        self.w = self.w - (learning_rate * self.dw)
        self.b = self.b - (learning_rate * self.db)








S=Simple_Neural_Network()
S.train_data_download()
F_L = Layer('relu')
S_L = Layer('relu', F_L.output, 64)
O_L = Layer('softmax', S_L.output, 10)
S.layers_collect(F_L)
S.layers_collect(S_L)
S.layers_collect(O_L)
S.train()
S.predict()