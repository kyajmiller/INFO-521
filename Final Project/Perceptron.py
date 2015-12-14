from __future__ import division
import pandas
import numpy


class Perceptron:
    ''' Implementation of the multinomial perceptron '''

    def __init__(self, numClasses=3, epochs=10, learningRate=1):
        self.numClasses = numClasses
        self.epochs = epochs
        self.learningRate = learningRate

        self.isTrained = False
        self.weights = None

    def train(self, trainingVectors, trainingLabels):
        ''' Learn the weights from the training data '''
        if self.isTrained:
            print("This perceptron is already trained")
        else:
            # initialize the weights
            self.weights = [numpy.transpose(numpy.matrix(numpy.zeros(trainingVectors.shape[1]))) for i in
                            xrange(self.numClasses)]
            adjustedWeights = [numpy.transpose(numpy.matrix(numpy.zeros(trainingVectors.shape[1]))) for i in
                               xrange(self.numClasses)]
            iterations = 0

            for i in xrange(self.epochs):
                errors = 0

                for j, label in zip(range(trainingVectors.shape[0]), trainingLabels):

                    v = trainingVectors[j, :]

                    p = self.predict(v)

                    if not p == label:  # If the prediction is wrong
                        for i in xrange(self.numClasses):
                            if i == p:
                                self.weights[i] -= self.learningRate * numpy.transpose(v)
                            else:
                                self.weights[i] += self.learningRate * numpy.transpose(v)
                        errors += 1

                    # Accumulate the w vectors
                    for i in range(len(adjustedWeights)):
                        adjustedWeights[i] += self.weights[i]

                    iterations += 1

                if errors == 0:
                    break
            # Perform the averaging
            for i in range(len(adjustedWeights)):
                adjustedWeights[i] /= iterations

            self.weights = adjustedWeights
            self.isTrained = True

    def predict(self, v):
        ''' Multiclass prediction '''
        p = [numpy.dot(v, w)[0] for w in self.weights]

        return p.index(max(p))


'''
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
            adjustedWeights = [numpy.transpose(numpy.matrix(numpy.zeros(trainingVectors.shape[1]))) for i in
                               xrange(self.numClasses)]
            iterations = 0

            for i in xrange(self.epochs):
                errors = 0
                for j, label in zip(range(trainingVectors.shape[0]), trainingLabels):
                    value = trainingVectors[j, :]
                    prediction = self.predict(value)

                    if prediction != label:
                        # if the prediction is wrong, adjust the weights accordingly
                        for k in xrange(self.numClasses):
                            if k == prediction:
                                self.weights[k] -= self.learningRate * numpy.transpose(value)
                            else:
                                self.weights[k] += self.learningRate * numpy.transpose(value)
                        errors += 1

                    for l in range(len(adjustedWeights)):
                        adjustedWeights[l] += self.weights[l]

                    iterations += 1

                if errors == 0:
                    break

            # average values
            for m in range(len(adjustedWeights)):
                adjustedWeights[m] /= iterations




    def predict(self, value):
        prediction = [numpy.dot(value, weight)[0] for weight in self.weights]
        maxValue = prediction.index(max(prediction))
        return maxValue
'''
