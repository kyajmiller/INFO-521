from __future__ import division
import numpy


class LogisticRegression(object):
    def __init__(self, trainingVectors, trainingLabels, testingVectors, testingLabels):
        self.trainingVectors = trainingVectors
        self.trainingLabels = trainingLabels
        self.testingVectors = testingVectors
        self.testingLabels = testingLabels
        self.n = self.trainingLabels.shape[0]

        self.betas = numpy.zeros(self.trainingVectors.shape[1])

    def sigmoid(self, x):
        y = 1 / (1 + numpy.exp(-x))
        return y

    def negativeLikelihood(self, betas):
        likelihood = 0
        for i in range(self.n):
            likelihood +=
