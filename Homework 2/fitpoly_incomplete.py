# Solution for ISTA 421 / INFO 521 Fall 2015, HW 2, Problem 1
# Author: Clayton T. Morrison, 12 September 2015

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Utilities


def read_data(filepath, d = ','):
    """ returns an np.array of the data """
    return np.genfromtxt(filepath, delimiter=d, dtype=None)


def plot_data(x, t):
    """
    Plot single input feature x data with corresponding response
    values t as a scatter plot
    :param x: sequence of 1-dimensional input data features
    :param t: sequence of 1-dimensional responses
    :return: None
    """
    plt.figure()  # Create a new figure object for plotting
    plt.scatter(x, t, edgecolor='b', color='w', marker='o')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Data')
    plt.pause(.1)  # required on some systems so that rendering can happen


def plot_model(x, w):
    """
    Plot the curve for an n-th order polynomial model:
        t = w0*x^0 + w1*x^1 + w2*x^2 + ... wn*x^n
    This works by creating a set of x-axis (plotx) points and
    then use the model parameters w to determine the corresponding
    t-axis (plott) points on the model curve.
    :param x: sequence of 1-dimensional input data features
    :param w: n-dimensional sequence of model parameters: w0, w1, w2, ..., wn
    :return: the plotx and plott values for the plotted curve
    """
    # NOTE: this assumes a figure() object has already been created.

    # plotx represents evenly-spaced set of 100 points on the x-axis
    # used for creating a relatively "smooth" model curve plot.
    # Includes points a little before the min x input (-0.25)
    # and a little after the max x input (+0.25)
    plotx = np.linspace(min(x)-0.25, max(x)+0.25, 100)

    # plotX (note that python is case sensitive, so this is not
    # the same as plotx with a lower-case x) is the "design matrix"
    # for our model curve inputs represented in plotx.
    # We need to do the same computation as we do when doing
    # model fitting (as in fitpoly(), below), except that we
    # don't need to infer (by the normal equations) the values
    # of w, as they're given here as input.
    # plotx.shape[0] ensures we create a matrix with the number of
    # rows corresponding to the number of points in plotx (this will
    # still work even if we change the number of plotx points to
    # something other than 100)
    plotX = np.zeros((plotx.shape[0], w.size))

    # populate the design matrix X
    for k in range(w.size):
        plotX[:, k] = np.power(plotx, k)

    # Take the (dot) inner product of the design matrix and the
    # parameter vector
    plott = np.dot(plotX, w)

    # plot the x (plotx) and t (plott) values in red
    plt.plot(plotx, plott, color='r', linewidth=2)

    plt.pause(.1)  # required on some systems so that rendering can happen
    return plotx, plott


def scale01(x):
    """
    HELPER FUNCTION: only needed if you are working with large
    x values.  This is NOT needed for problems 1, 2 and 4.

    The values of x could be arbitrary.  The math does not care
    about their magnitude, but computationally, we need to be
    careful here as we are taking powers of the values of x;
    if values of x are large, then taking large powers of x
    might exceed what can be represented numerically.
    This function scales the input data to be the range [0, 1]
    (i.e., between 0 and 1, inclusive)
    :param x: sequence of 1-dimensional input data features
    :return: x values linearly scaled to range [0, 1]
    """
    x_min = min(x)
    x_range = max(x) - x_min
    return (x - x_min) / x_range


# -------------------------------------------------------------------------
# fitpoly

def fitpoly(x, t, model_order):
    """
    Given "training" data in input feature sequence x and
    corresponding target value sequence t, and a specified
    polynomial of order model_order, determine the linear
    least mean squared (LMS) error best fit for parameters w,
    using the generalized matrix normal equation.
    model_order is a non-negative integer, n, representing the
    highest polynomial order term of the polynomial model:
        t = w0*x^0 + w1*x^1 + w2*x^2 + ... wn*x^n
    :param x: sequence of 1-dimensional input data features
    :param t: sequence of target response values
    :param model_order: integer representing the maximum order of the polynomial model
    :return: parameter vector w
    """

    # Construct the empty design matrix
    # np.zeros takes a tuple representing the number of
    # rows and columns, (rows,columns), filled with zeros.
    # The number of columns is model_order+1 because a model_order
    # of 0 requires one column (filled with input x values to the
    # power of 0), model_order=1 requires two columns (first input x
    # values to power of 0, then column of input x values to power 1),
    # and so on...
    X = np.zeros((x.shape[0], model_order+1))
    # Fill each column of the design matrix with the corresponding
    for k in range(model_order+1):  # w.size
        X[:, k] = np.power(x, k)

    #### YOUR CODE HERE ####
    w = None  # Calculate w vector (as an np.array)
    
    return w


# -------------------------------------------------------------------------
# Script to run on particular data set
# -------------------------------------------------------------------------

data_path = '../data/womens100.csv'       ## Problem 2
# data_path = '../data/synthdata2015.csv'   ## Problem 4

# -----------
# The following data is provided just for fun, not used in HW 2.
# This is the data for the men's 100, which has been the recurring
# example in the class
# data_path = '../data/mens100.csv'

model_order = 1  # for problem 2
# model_order = 3  # for problem 4

data = read_data(data_path, ',')

# x = scale01(Data[:, 0])  # extract x (slice first column) and scale so x \in [0,1]
x = data[:, 0]  # extract x (slice first column)
t = data[:, 1]  # extract t (slice second column)

plot_data(x, t)
w = fitpoly(x, t, model_order)

print 'Identified model parameters w:', w

plot_model(x, w)

plt.show()  # generally only needed if you don't use plt.ion()

# Uncomment this last line if your matplotlib window is not staying open
# raw_input('Press <ENTER> to quit...')
