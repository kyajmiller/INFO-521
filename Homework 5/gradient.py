import numpy as np
import stacked_autoencoder
import utils_hw


# this function accepts a 2D vector as input.
# Its outputs are:
#   value: h(x1, x2) = x1^2 + 3*x1*x2
#   grad: A 2x1 vector that gives the partial derivatives of h with respect to x1 and x2
# Note that when we pass simple_quadratic_function(x) to computeNumericalGradients, we're assuming
# that computeNumericalGradients will use only the first returned value of this function.
def simple_quadratic_function(x):
    value = x[0] ** 2 + 3 * x[0] * x[1]

    grad = np.zeros(shape=2, dtype=np.float32)
    grad[0] = 2 * x[0] + 3 * x[1]
    grad[1] = 3 * x[0]

    return value, grad


# theta: a vector of parameters
# J: a function that outputs a real-number. Calling y = J(theta) will return the
# function value (the activation of the NN "neuron") at theta.
# The return value, gradient, should be a numpy array of length theta,
# holding a floating value representing the gradient for each theta parameter
def compute_gradient(theta, hidden_size, visible_size, data, a2, delta_a3, delta_a2):
    # I decided to change the arguments of this function because I really don't see the need for having a
    # separate function for this at all. In order to get what I need to calculate the gradient from just the theta,
    # I'd literally end up doing the same functions as in utils_hw.sparse_autoencoder_cost.
    # This function now takes arguments of theta, hidden_size, visible_size, data, a2, delta3, delta2
    epsilon = 0.0001

    ### YOUR CODE HERE ###
    w1EndPoint = hidden_size * visible_size
    w2EndPoint = w1EndPoint * 2
    b1EndPoint = w2EndPoint + hidden_size

    w1 = theta[:w1EndPoint].reshape(hidden_size, visible_size)
    w2 = theta[w1EndPoint:w2EndPoint].reshape(visible_size, hidden_size)
    b1 = theta[w2EndPoint:b1EndPoint]
    b2 = theta[b1EndPoint:]

    w1_gradient = np.dot(delta_a2, np.transpose(data))
    w1_gradient = w1_gradient / data.shape[1] + epsilon * w1

    w2_gradient = np.dot(delta_a3, np.transpose(a2))
    w2_gradient = w2_gradient / data.shape[1] + epsilon * w2

    b1_gradient = np.sum(delta_a2, axis=1)
    b1_gradient = b1_gradient / data.shape[1]

    b2_gradient = np.sum(delta_a3, axis=1)
    b2_gradient = b2_gradient / data.shape[1]

    w1_gradient = np.array(w1_gradient)
    w2_gradient = np.array(w2_gradient)
    b1_gradient = np.array(b1_gradient)
    b2_gradient = np.array(b2_gradient)

    gradient = np.concatenate(
        (w1_gradient.flatten(), w2_gradient.flatten(), b1_gradient.flatten(), b2_gradient.flatten()))

    return gradient


# This code can be used to check your numerical gradient implementation
# in computeNumericalGradient.m
# It analytically evaluates the gradient of a very simple function called
# simpleQuadraticFunction (see below) and compares the result with your numerical
# solution. Your numerical gradient implementation is incorrect if
# your numerical solution deviates too much from the analytical solution.
def check_gradient():
    x = np.array([4, 10], dtype=np.float64)
    (value, grad) = simple_quadratic_function(x)

    num_grad = compute_gradient(simple_quadratic_function, x)
    print num_grad, grad
    print "The above two columns you get should be very similar.\n" \
          "(Left-Your Numerical Gradient, Right-Analytical Gradient)\n"

    diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
    print diff
    print "Norm of the difference between numerical and analytical num_grad (should be < 1e-9)\n"


def check_stacked_autoencoder():
    """
    # Check the gradients for the stacked autoencoder
    #
    # In general, we recommend that the creation of such files for checking
    # gradients when you write new cost functions.
    #

    :return:
    """
    ## Setup random data / small model

    input_size = 64
    hidden_size_L1 = 36
    hidden_size_L2 = 25
    lambda_ = 0.01
    data = np.random.randn(input_size, 10)
    labels = np.random.randint(4, size=10)
    num_classes = 4

    stack = [dict() for i in range(2)]
    stack[0]['w'] = 0.1 * np.random.randn(hidden_size_L1, input_size)
    stack[0]['b'] = np.random.randn(hidden_size_L1)
    stack[1]['w'] = 0.1 * np.random.randn(hidden_size_L2, hidden_size_L1)
    stack[1]['b'] = np.random.randn(hidden_size_L2)
    softmax_theta = 0.005 * np.random.randn(hidden_size_L2 * num_classes)

    params, net_config = stacked_autoencoder.stack2params(stack)

    stacked_theta = np.concatenate((softmax_theta, params))

    cost, grad = stacked_autoencoder.stacked_autoencoder_cost(stacked_theta, input_size,
                                                              hidden_size_L2, num_classes,
                                                              net_config, lambda_, data, labels)

    # Check that the numerical and analytic gradients are the same
    J = lambda x: stacked_autoencoder.stacked_autoencoder_cost(x, input_size, hidden_size_L2,
                                                               num_classes, net_config, lambda_,
                                                               data, labels)
    num_grad = compute_gradient(J, stacked_theta)

    print num_grad, grad
    print "The above two columns you get should be very similar.\n" \
          "(Left-Your Numerical Gradient, Right-Analytical Gradient)\n"

    diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
    print diff
    print "Norm of the difference between numerical and analytical num_grad (should be < 1e-9)\n"
