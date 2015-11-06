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
    r = (1 - a - y)/(1 - a - (2 * y) + B + N)
    var = (-math.pow(r, 2) * (math.pow((1 - r), 2))) / ((math.pow((1 - r), 2) * (1 - a - y)) - (math.pow(r, 2) * (B - N + y)))

    betaDistribution = beta()
    normalDistribution = norm()

    print(beta)


plotFirstSet()
