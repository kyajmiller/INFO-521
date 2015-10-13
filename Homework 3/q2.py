__author__ = 'Kya'
import math


def calculateX(x):
    solution = 60 + (0.1 * x) + (0.5 * math.pow(x, 3)) + (0.05 * math.pow(x, 4))
    return solution


total = 0
for i in range(16):
    x = i - 10
    total += calculateX(x)

print(total / 15)
