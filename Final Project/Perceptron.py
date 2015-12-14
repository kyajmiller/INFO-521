from __future__ import division
import numpy


class Perceptron(object):
    def __init__(self, numClasses=2, epochs=10, learningRate=1.5):
        self.numClasses = numClasses  # how many classes, this absolutely needs to match the number of unique labels used
        self.epochs = epochs  # how many times to run the process of adjusting the weights via the learningRate
        self.learningRate = learningRate  # the amount by which the weights are adjusted each time

        self.isTrained = False
        self.weights = None

    def train(self, trainingVectors, trainingLabels):
        # get the weights

        if self.isTrained:
            print("Perceptron is already trained!")
        else:
            # initialize the weights
            self.weights = [numpy.transpose(numpy.matrix(numpy.zeros(trainingVectors.shape[1]))) for i in
                            xrange(self.numClasses)]
            adjustedWeights = [numpy.transpose(numpy.matrix(numpy.zeros(trainingVectors.shape[1]))) for i in
                               xrange(self.numClasses)]

            # keep track of how many iterations were run so we can average the values of the adjusted weights
            numIterations = 0

            for i in xrange(self.epochs):
                errors = 0

                for j, label in zip(range(trainingVectors.shape[0]), trainingLabels):
                    # make a prediction based on current weights
                    value = trainingVectors[j, :]
                    prediction = self.predict(value)

                    if prediction != label:
                        # if prediction is wrong, adjust weights accordingly
                        for k in xrange(self.numClasses):
                            if k == prediction:
                                self.weights[k] -= self.learningRate * numpy.transpose(value)
                            else:
                                self.weights[k] += self.learningRate * numpy.transpose(value)
                        errors += 1

                    # get weight vectors
                    for l in range(len(adjustedWeights)):
                        adjustedWeights[l] += self.weights[l]

                    numIterations += 1

                # if there's no errors, then it's done
                if errors == 0:
                    break

            # do averaging
            for m in range(len(adjustedWeights)):
                adjustedWeights[m] /= numIterations

            self.weights = adjustedWeights
            self.isTrained = True

    def predict(self, value):
        # return the index of the maximum predicted value
        prediction = [numpy.dot(value, w)[0] for w in self.weights]
        maxPrediction = prediction.index(max(prediction))
        return maxPrediction
