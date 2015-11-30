# This is a set of utilities to run the NN excersis in ISTA 421, Introduction to ML
# By Leon F. Palafox, December, 2014

from __future__ import division
import numpy as np
import math
import gradient


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def initialize(hidden_size, visible_size):
    # choose weights uniformly from the interval [-r, r] following what we saw in class

    ### YOUR CODE HERE ###
    r = math.sqrt(6 / (hidden_size + visible_size + 1))
    rand = np.random.RandomState()
    w1 = np.asarray(rand.uniform(-r, r, (hidden_size, visible_size)))
    w2 = np.asarray(rand.uniform(-r, r, (visible_size, hidden_size)))

    b1 = np.zeros(hidden_size)
    b2 = np.zeros(visible_size)

    theta = np.concatenate((w1.flatten(), w2.flatten(), b1.flatten(), b2.flatten()))

    return theta


def sparse_autoencoder_cost(theta, visible_size, hidden_size,
                            lambda_, data):
    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    ### YOUR CODE HERE ###

    w1 = theta[:hidden_size*visible_size].reshape(hidden_size, visible_size)
    w2 = theta[hidden_size*visible_size:(hidden_size*visible_size)*2].reshape(visible_size, hidden_size)
    b1 = theta[(hidden_size*visible_size)*2:((hidden_size*visible_size)*2)+hidden_size]
    b2 = theta[((hidden_size*visible_size)*2)+hidden_size:]

    MNumTrainingExamples = data.shape[1]

    z2 = np.dot(w1, data) + np.transpose(np.tile(b1, (MNumTrainingExamples, 1)))

    a2 = sigmoid(z2)

    z3 = np.dot(w2, a2) + np.transpose(np.tile(b2, (MNumTrainingExamples, 1)))

    a3 = sigmoid(z3)

    JsumOfSquaredError = (1 / 2) * np.sum(np.power((a3 - data), 2)) / data.shape[1]

    weightDecay = (lambda_ / 2) * (np.sum(np.power(w1, 2)) + np.sum(np.power(w2, 2)))
    cost = JsumOfSquaredError + weightDecay

    delta_a3 = np.multiply(-(data - a3), sigmoid_prime(z3))
    delta_a2 = np.multiply(np.dot(np.transpose(w2), delta_a3), sigmoid_prime(z2))

    w1_gradient = (np.multiply((np.dot(delta_a2, np.transpose(data)) / MNumTrainingExamples + lambda_), w1)).flatten()
    w2_gradient = (np.multiply((np.dot(delta_a3, np.transpose(a2)) / MNumTrainingExamples + lambda_), w2)).flatten()
    b1_gradient = (np.sum(delta_a2, axis=1) / MNumTrainingExamples).flatten()
    b2_gradient = (np.sum(delta_a3, axis=1) / MNumTrainingExamples).flatten()

    grad = np.concatenate((w1_gradient, w2_gradient, b1_gradient, b2_gradient))

    return cost, grad


# visible_size: the number of input units (probably 64)
# hidden_size: the number of hidden units (probably 25)
# lambda_: weight decay parameter
# sparsity_param: The desired average activation for the hidden units (denoted in the lecture
#                            notes by the greek alphabet rho, which looks like a lower-case "p").
# beta: weight of sparsity penalty term
# data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.
#
# The input theta is a vector (because minFunc expects the parameters to be a vector).
# We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
# follows the notation convention of the lecture notes.
# Returns: (cost,gradient) tuple


def sparse_autoencoder(theta, hidden_size, visible_size, data):
    """
    :param theta: trained weights from the autoencoder
    :param hidden_size: the number of hidden units (probably 25)
    :param visible_size: the number of input units (probably 64)
    :param data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example.
    """

    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.
    W1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    b1 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]

    # Number of training examples
    m = data.shape[1]

    # Forward propagation
    z2 = W1.dot(data) + np.tile(b1, (m, 1)).transpose()
    a2 = sigmoid(z2)
    return a2
