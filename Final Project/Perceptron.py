import pandas
import numpy


class Perceptron(object):
    def __init__(self, numClasses=2, epochs=10, learningRate=1):
        self.numClasses = numClasses
        self.epochs = epochs
        self.learningRate = learningRate

        self.isTrained = False
