from __future__ import division
import numpy


class LogisticRegression(object):
    def __init__(self, trainingVectors, trainingLabels, testingVectors, testingLabels):
        self.trainingVectors = trainingVectors
        self.trainingLabels = trainingLabels
        self.testingVectors = testingVectors
        self.testingLabels = testingLabels
        self.n = self.trainingLabels.shape[0]

        self.alpha = 0.1
        self.betas = numpy.zeros(self.trainingVectors.shape[1])

    def sigmoid(self, x):
        y = 1 / (1 + numpy.exp(-x))
        return y

    def likelihood(self, betas):
        likelihood = 0
        for i in range(self.n):
            likelihood += numpy.log(self.sigmoid(self.trainingLabels[i] * numpy.dot(betas, self.trainingVectors[i, :])))

        for j in range(1, self.trainingVectors.shape[1]):
            likelihood -= (self.alpha / 2) * numpy.power(self.betas[j], 2)

        return likelihood
