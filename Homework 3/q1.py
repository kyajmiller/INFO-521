"""
Kya Miller
INFO 521 Homework 3
p(y) = ((lambda^y)/(y.factorial)) * e^-lambda
lambda = 8
"""
import math


def calculateProbability(y):
    lamb = 8
    e = 10
    probability = (math.pow(lamb, y) / math.factorial(y)) * math.pow(e, -lamb)
    return probability


probYLessThanEqualToSix = 0
for i in range(6):
    y = i + 1
    probYLessThanEqualToSix += calculateProbability(y)

probYGreaterThanSix = 1 - probYLessThanEqualToSix

print(probYLessThanEqualToSix)
print(probYGreaterThanSix)
