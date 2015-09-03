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

numpy.random.seed(seed=2)
