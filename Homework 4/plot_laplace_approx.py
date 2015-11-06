from __future__ import division
from scipy.stats import beta
from scipy.stats import norm
import numpy
import math

import matplotlib.pyplot as plt


def plotFirstSet():
    a = 5
    B = 5
    N = 20
    y = 10

    plotBetaPosteriorAndLaplaceApproximation(a, B, N, y)


def plotBetaPosteriorAndLaplaceApproximation(a, B, N, y):
    # r = (1 - a - y) / (-N - B - a + 2)
    # var = (math.pow(r, 2) * math.pow((1 - r), 2)) / ((math.pow((1 - r), 2) * (1 - a - y)) + (math.pow(r, 2) * (1 - B - N + y)))
    # r = (1 - a - y)/(1 - a - (2 * y) + B + N)
    # var = (-math.pow(r, 2) * math.pow((1 - r), 2)) / ((math.pow((1 - r), 2) * (1 - a - y)) + (math.pow(r, 2) * (1 - B - N + y)))

    r = (1 - y - a) / (2 - a - N - B)
    var = (-math.pow(r, 2) * math.pow((1 - r), 2)) / (
    (math.pow((1 - r), 2) * (1 - a - y)) + (math.pow(r, 2) * (1 - B - N + y)))
    # var = (math.pow(r, 2) * math.pow((1 - r), 2)) / ((math.pow((1 - r), 2) * (y + a - 1)) + (math.pow(r, 2) * (1 - N + y - B)))
    x = numpy.linspace(0.01, 1, 1000)

    print(r)
    print(var)

    plotBeta = beta.pdf(x, a, B, loc=0, scale=1)
    plt.plot(x, plotBeta)
    plt.show()

    '''
    t = beta.pdf(x, a, B)
    plt.plot(x, t)

    t = beta.pdf(x, a, B, loc=r, scale=math.sqrt(var))
    plt.plot(x, t, 'r--')
    plt.show()


    laplace = beta.pdf(x, a, B, loc=r, scale=math.sqrt(var))
    plt.plot(x, laplace, 'r')
    plt.show()

    print(math.sqrt(var))
    print(r)
    '''





plotFirstSet()
