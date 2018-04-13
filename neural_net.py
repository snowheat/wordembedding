from sklearn import neural_network, linear_model, naive_bayes
import numpy as np


class NeuralNet:
    def __init__(self):
        self.__model = type('test', (object,), {})()
        pass

    def train(self, data, label):
        self.__model = neural_network.MLPClassifier()
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
