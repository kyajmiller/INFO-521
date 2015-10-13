__author__ = 'Kya'
# approx_expected_value.py
# Approximating expected values via sampling
import numpy as np
import matplotlib.pyplot as plt


# We are trying to estimate the expected value of
# $f(x) = 60 + 0.1x + 0.5x^3 + 0.05x^4$
##
# ... where
# $p(x)=U(-10,5)$
##
# ... which is given by:
# $\int 60 + 0.1x + 0.5x^3 + 0.05x^4 p(x) dx$
##
# The analytic result is:
# $\frac{871.6}{15}$ = 58.11...

# Sample 5000 uniformly random values in [0..1]
xs = np.random.uniform(low=0.0, high=1.0, size=5000)
# compute the expectation of x, where f(x) = 60 + 0.1x + 0.5x^3 + 0.05x^4
expX = np.mean((60 + (0.1 * xs) + (0.5 * np.power(xs, 3)) + (0.05 * np.power(xs, 4))))
print '\nSample-based approximation: {:f}'.format(expX)

# Store the evolution of the approximation, every 10 samples
sample_sizes = np.arange(1, xs.shape[0], 10)
expX_evol = np.zeros((sample_sizes.shape[0]))  # storage for the evolving estimate...
# the following computes the mean of the sequence up to i, as i iterates
# through the sequence, storing the mean in expX_evol:
for i in range(sample_sizes.shape[0]):
    expX_evol[i] = np.mean((60 + (0.1 * xs[0:sample_sizes[i]]) + (0.5 * np.power(xs[0:sample_sizes[i]], 3)) + (
    0.05 * np.power(xs[0:sample_sizes[i]], 4))))

# Create plot of evolution of the approximation
plt.figure()
# plot the curve of the estimation of the expected value of f(x)=y^2
plt.plot(sample_sizes, expX_evol)
# The true, analytic result of the expected value of f(y)=y^2 where y ~ U(0,1): $\frac{1}{3}$
# plot the analytic expected result as a red line:
plt.plot(np.array([sample_sizes[0], sample_sizes[-1]]), np.array([1. / 3, 1. / 3]), color='r')
plt.xlabel('Sample size')
plt.ylabel('Approximation of expectation')
plt.title('Approximation of expectation of $f(y) = y^2$')
plt.pause(.1)  # required on some systems so that rendering can happen

plt.show()  # keeps the plot open
