from sklearn import neural_network, linear_model, naive_bayes
import numpy as np


class NeuralNet:
    def __init__(self):
        self.__model = type('test', (object,), {})()
        pass

    def train(self, data, label):
        self.__model = neural_network.MLPClassifier(hidden_layer_sizes=100, max_iter=100, alpha=0.2,
                                                    learning_rate_init=0.2)
        # self.__model = linear_model.SGDClassifier()

        # self.__model = naive_bayes.GaussianNB()
        self.__model = self.__model.fit(data, label)

        # predicted_y = self.__model.predict(data)
        # print(np.mean(predicted_y == data))
        pass

    def test(self, data, label):
        predicted_y = self.__model.predict(data)

        print(np.mean(predicted_y == label))
        pass
