import pandas
import numpy


class Perceptron(object):
    def __init__(self, numClasses=2, epochs=10, learningRate=1):
        self.numClasses = numClasses
        self.epochs = epochs
        self.learningRate = learningRate

        self.isTrained = False
        self.weights = None

    def train(self, trainingVectors, trainingLabels):
        if self.isTrained:
            print("Perceptron is already trained!")
        else:
            self.weights = [numpy.transpose(numpy.matrix(numpy.zeros(trainingVectors.shape[1]))) for i in
                            xrange(self.numClasses)]
            actualWeights = [numpy.transpose(numpy.matrix(numpy.zeros(trainingVectors.shape[1]))) for i in
                             xrange(self.numClasses)]
            iterations = 0

            for i in xrange(self.epochs):
                errors = 0
                for j, label in zip(range(trainingVectors.shape[0]), trainingLabels):
                    value = trainingVectors[j, :]
                    prediction = self.predict(value)

    def predict(self, value):
        prediction = [numpy.dot(value, weight)[0] for weight in self.weights]
        maxValue = prediction.index(max(prediction))
        return maxValue
