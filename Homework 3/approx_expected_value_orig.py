# approx_expected_value.py
# Port of approx_expected_value.m
# From A First Course in Machine Learning, Chapter 2.
# Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# Approximating expected values via sampling
import numpy as np
import matplotlib.pyplot as plt


# We are trying to estimate the expected value of
# $f(y) = y^2$
##
# ... where
# $p(y)=U(0,1)$
## 
# ... which is given by:
# $\int y^2 p(y) dy$
##
# The analytic result is:
# $\frac{1}{3}$ = 0.333...
# (NOTE: this just gives you the analytic result -- you should be abld to derive it!)

# Sample 100 uniformly random values in [0..1]
ys = np.random.uniform(low=0.0, high=1.0, size=100)
# compute the expectation of y, where y is the function that squares its input
ey2 = np.mean(np.power(ys, 2))
print '\nSample-based approximation: {:f}'.format(ey2)

# Store the evolution of the approximation, every 10 samples
sample_sizes = np.arange(1, ys.shape[0], 10)
ey2_evol = np.zeros((sample_sizes.shape[0]))  # storage for the evolving estimate...
# the following computes the mean of the sequence up to i, as i iterates 
# through the sequence, storing the mean in ey2_evol:
for i in range(sample_sizes.shape[0]):
    ey2_evol[i] = np.mean(np.power(ys[0:sample_sizes[i]], 2))

# Create plot of evolution of the approximation
plt.figure()
# plot the curve of the estimation of the expected value of f(x)=y^2
plt.plot(sample_sizes, ey2_evol)
# The true, analytic result of the expected value of f(y)=y^2 where y ~ U(0,1): $\frac{1}{3}$
# plot the analytic expected result as a red line:
plt.plot(np.array([sample_sizes[0], sample_sizes[-1]]), np.array([1. / 3, 1. / 3]), color='r')
plt.xlabel('Sample size')
plt.ylabel('Approximation of expectation')
plt.title('Approximation of expectation of $f(y) = y^2$')
plt.pause(.1)  # required on some systems so that rendering can happen

plt.show()  # keeps the plot open
