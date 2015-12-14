from __future__ import division
from scipy.optimize.optimize import fmin_cg, fmin_bfgs, fmin
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

    def negativeLikelihood(self, betas):
        return -self.likelihood(betas)

    def train(self):
        dBk = lambda B, k: (k > 0) * self.alpha * B[k] - numpy.sum([self.trainingLabels[i] * self.trainingVectors[
            i, k] * self.sigmoid(-self.trainingLabels[i] * numpy.dot(B, self.trainingVectors[i, :])) for i in
                                                                    range(self.n)])

        dB = lambda B: numpy.array([dBk(B, j) for j in range(self.trainingVectors.shape[1])])

        self.betas = fmin_bfgs(self.negativeLikelihood(self.betas), self.betas, fprime=dB)
