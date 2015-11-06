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

    plt.figure(1)
    plotBetaPosteriorAndLaplaceApproximation(a, B, N, y)


def plotSecondSet():
    a = 3
    B = 15
    N = 10
    y = 3

    plt.figure(2)

    plotBetaPosteriorAndLaplaceApproximation(a, B, N, y)


def plotThirdSet():
    a = 1
    B = 30
    N = 10
    y = 3

    plt.figure(3)
    plotBetaPosteriorAndLaplaceApproximation(a, B, N, y)


def plotBetaPosteriorAndLaplaceApproximation(a, B, N, y):
    r = (1 - y - a) / (2 - a - N - B)
    var = -(math.pow(r, 2) * math.pow((1 - r), 2)) / (
        (math.pow((1 - r), 2) * (1 - a - y)) + (math.pow(r, 2) * (1 - B - N + y)))
    x = numpy.linspace(0.01, 1, 1000)

    print(r)
    print(var)

    plt.title("True Beta Posterior & Laplace Approximation (a=%s, B=%s, N=%s, y=%s)" % (a, B, N, y))
    plt.xlabel("r")
    plt.ylabel("p(r|y)")

    betaT = beta.pdf(x, a, B)
    plt.plot(x, betaT, 'b')

    normT = norm.pdf(x, loc=r, scale=math.sqrt(var))
    plt.plot(x, normT, 'r--')







plotFirstSet()
plotSecondSet()
plotThirdSet()
plt.show()
