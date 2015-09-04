__author__ = 'Kya'
import numpy

numpy.random.seed(seed=1)
a = numpy.random.rand(3, 1)
b = numpy.random.rand(3, 1)

aTranspose = numpy.transpose(a)

print(a)
print(b)

plus = a + b
elementWiseMultiply = a * b
dotPointMultiply = aTranspose * b

print(plus)
print(elementWiseMultiply)
print(dotPointMultiply)

numpy.random.seed(seed=2)
X = numpy.random.rand(3, 3)

aTransposeX = aTranspose * X
aTransposeXb = aTranspose * X * b
Xinverse = numpy.linalg.inv(X)

print(aTransposeX)
print(aTransposeXb)
print(Xinverse)
