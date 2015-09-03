__author__ = 'Kya'
import numpy
import matplotlib.pyplot as plt

plt.ylabel('sin(x)')
plt.xlabel('x values')
plt.title('Sine Function for x from 0.0 to 10.0')

xValues = numpy.arange(0, 10, 0.01)
yValues = numpy.sin(xValues)

plt.figure(1)
plt.plot(xValues, yValues)
plt.show()
