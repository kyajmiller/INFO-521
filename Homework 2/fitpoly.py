__author__ = 'Kya'
import numpy as np
import matplotlib.pyplot as plt


def read_data(filepath, d = ','):
    return np.genfromtxt(filepath, delimiter=d, dtype=None)


def plot_data(x, t):
    plt.figure()  # Create a new figure object for plotting
    plt.scatter(x, t, edgecolor='b', color='w', marker='o')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Data')
    plt.pause(.1)


def plot_model(x, w):
    plotx = np.linspace(min(x)-0.25, max(x)+0.25, 100)

    plotX = np.zeros((plotx.shape[0], w.size))

    for k in range(w.size):
        plotX[:, k] = np.power(plotx, k)

    plott = np.dot(plotX, w)

    plt.plot(plotx, plott, color='r', linewidth=2)

    plt.pause(.1)
    return plotx, plott


def scale01(x):
    x_min = min(x)
    x_range = max(x) - x_min
    return (x - x_min) / x_range


def fitpoly(x, t, model_order):
    X = np.zeros((x.shape[0], model_order+1))
    for k in range(model_order+1):  # w.size
        X[:, k] = np.power(x, k)

    w = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), t)
    return w


# -------------------------------------------------------------------------
# Script to run on particular data set
# -------------------------------------------------------------------------

data_path = 'womens100.csv' ##Problem 2
# data_path = 'synthdata2015.csv'   ## Problem 4

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