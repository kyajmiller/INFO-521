# Solution for ISTA 421 / INFO 521 Fall 2015, HW 2, Problem 5
# Author: Clayton T. Morrison, 13 September 2015
# Based on cv_demo.m
# From A First Course in Machine Learning, Chapter 1.
# Simon Rogers, 31/10/11 [simon.rogers@glasgow.ac.uk]

# NOTE: In its released form, this script will NOT run
#       You will get a syntax error on line 79 because w has not been defined

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Utilities


def random_permutation_matrix(n):
    """
    Generate a permutation matrix: an NxN matrix in which each row
    and each column has only one 1, with 0's everywhere else.
    See: https://en.wikipedia.org/wiki/Permutation_matrix
    :param n: size of the square permutation matrix
    :return: NxN permutation matrix
    """
    rows = np.random.permutation(n)
    cols = np.random.permutation(n)
    m = np.zeros((n, n))
    for r, c in zip(rows, cols):
        m[r][c] = 1
    return m


def permute_rows(X, P=None):
    """
    Permute the rows of a 2-d array (matrix) according to
    permutation matrix P.
    If no P is provided, a random permutation matrix is generated.
    :param X: 2-d array
    :param P: Optional permutation matrix; default=None
    :return: new version of X with rows permuted according to P
    """
    if P is None:
        P = random_permutation_matrix(X.shape[0])
    return np.dot(P, X)


def permute_cols(X, P=None):
    """
    Permute the columns of a 2-d array (matrix) according to
    permutation matrix P.
    If no P is provided, a random permutation matrix is generated.
    :param X: 2-d array
    :param P: Optional permutation matrix; default=None
    :return: new version of X with columns permuted according to P
    """
    if P is None:
        P = random_permutation_matrix(X.shape[0])
    return np.dot(X, P)


# -------------------------------------------------------------------------
# Utilities from fitpoly


def read_data(filepath, d = ','):
    """ returns an np.array of the data """
    return np.genfromtxt(filepath, delimiter=d, dtype=None)


def plot_data(x, t):
    """
    Plot single input feature x data with corresponding response
    values t as a scatter plot
    :param x: sequence of 1-dimensional input data features
    :param t: sequence of 1-dimensional responses
    :return: None
    """
    plt.figure()  # Create a new figure object for plotting
    plt.scatter(x, t, edgecolor='b', color='w', marker='o')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Data')
    plt.pause(.1)  # required on some systems so that rendering can happen


def plot_model(x, w, color='r'):
    """
    Plot the curve for an n-th order polynomial model:
        t = w0*x^0 + w1*x^1 + w2*x^2 + ... wn*x^n
    This works by creating a set of x-axis (plotx) points and
    then use the model parameters w to determine the corresponding
    t-axis (plott) points on the model curve.
    :param x: sequence of 1-dimensional input data features
    :param w: n-dimensional sequence of model parameters: w0, w1, w2, ..., wn
    :return: the plotx and plott values for the plotted curve
    """
    # NOTE: this assumes a figure() object has already been created.
    plotx = np.linspace(min(x)-0.25, max(x)+0.25, 100)
    plotX = np.zeros((plotx.shape[0], w.size))
    for k in range(w.size):
        plotX[:, k] = np.power(plotx, k)
    plott = np.dot(plotX, w)
    plt.plot(plotx, plott, color=color, linewidth=2)
    plt.pause(.1)  # required on some systems so that rendering can happen
    return plotx, plott

# -------------------------------------------------------------------------
# Synthetic data generation


def generate_synthetic_data(N, w, xmin=-5, xmax=5, sigma=150):
    """
    """
    # generate N random input x points between [xmin, xmax]
    x = (xmax - xmin)*np.random.rand(N) + xmin

    # generate response with Gaussian random noise
    X = np.zeros((x.size, w.size))
    for k in range(w.size):
        X[:, k] = np.power(x, k)
    t = np.dot(X, w) + sigma*np.random.randn(x.shape[0])

    return x, t


def plot_synthetic_data(x, t, w, filepath=None):
    plot_data(x, t)
    plt.title('Plot of synthetic data; green curve is original generating function')
    plot_model(x, w, color='g')
    if filepath:
        plt.savefig(filepath, format='pdf')


# -------------------------------------------------------------------------


def plot_cv_results(train_loss, cv_loss, ind_loss, log_scale_p=False):
    plt.figure()
    if log_scale_p:
        plt.title('Log-scale Mean Square Error Loss')
        ylabel = 'Log MSE Loss'
    else:
        plt.title('Mean Squared Error Loss')
        ylabel = 'MSE Loss'

    x = np.arange(0, train_loss.shape[1])

    train_loss_mean = np.mean(train_loss, 0)
    cv_loss_mean = np.mean(cv_loss, 0)
    ind_loss_mean = np.mean(ind_loss, 0)

    # put y-axis on same scale for all plots
    min_ylim = min(list(train_loss_mean) + list(cv_loss_mean) + list(ind_loss_mean))
    min_ylim = int(np.floor(min_ylim))
    max_ylim = max(list(train_loss_mean) + list(cv_loss_mean) + list(ind_loss_mean))
    max_ylim = int(np.ceil(max_ylim))

    plt.subplot(131)
    plt.plot(x, train_loss_mean, linewidth=2)
    plt.xlabel('Model Order')
    plt.ylabel(ylabel)
    plt.title('Train Loss')
    plt.pause(.1) # required on some systems so that rendering can happen
    plt.ylim(min_ylim, max_ylim)

    plt.subplot(132)
    plt.plot(x, cv_loss_mean, linewidth=2)
    plt.xlabel('Model Order')
    plt.ylabel(ylabel)
    plt.title('CV Loss')
    plt.pause(.1) # required on some systems so that rendering can happen
    plt.ylim(min_ylim, max_ylim)

    plt.subplot(133)
    plt.plot(x, ind_loss_mean, linewidth=2)
    plt.xlabel('Model Order')
    plt.ylabel(ylabel)
    plt.title('Independent Test Loss')
    plt.pause(.1) # required on some systems so that rendering can happen
    plt.ylim(min_ylim, max_ylim)

    plt.subplots_adjust(right=0.95, wspace=0.4)
    plt.draw()


def run_cv( K, maxorder, x, t, testx, testt, randomize_data=False, title='CV' ):

    N = x.shape[0]  # number of data points

    # Use when you want to ensure the order of teh data has been
    # randomized before splitting into folds
    # Note that in the simple demo here, the data is already in
    # random order.
    if randomize_data:
        # use the same permutation on x and t!
        P = random_permutation_matrix(x.size)
        x = permute_rows(x, P)
        t = permute_rows(t, P)

    # Design matrix for training
    X = np.zeros((x.shape[0], maxorder+1))

    # Design matrix for independent test data
    testX = np.zeros((testx.shape[0], maxorder+1))

    # Create approximately equal-sized fold indices
    # These correspond to indices in the design matrix (X) rows
    # (where each row represents one training input x)
    fold_indices = map(lambda x: int(x), np.linspace(0, N, K+1))

    # storage for recording loss
    # rows = fold loss
    # columns = model polynomial order
    cv_loss = np.zeros((K, maxorder + 1))     # cross-validation loss
    train_loss = np.zeros((K, maxorder + 1))  # training loss
    ind_loss = np.zeros((K, maxorder + 1))    # independent test loss

    # iterate over model orders
    for p in range(maxorder + 1):
        X[:, p] = np.power(x, p)

        testX[:, p] = np.power(testx, p)

        # iterate over folds
        for fold in range(K):
            # Partition the data
            # foldX, foldt contains the data for just one fold being held out
            # trainX, traint contains all other data

            foldX = X[fold_indices[fold]:fold_indices[fold+1], 0:p+1]
            foldt = t[fold_indices[fold]:fold_indices[fold+1]]

            trainX = np.copy(X[:, 0:p+1])
            # remove the fold x from teh training set
            trainX = np.delete(trainX, np.arange(fold_indices[fold], fold_indices[fold+1]), 0)

            traint = np.copy(t)
            # remove the fold t from teh training set
            traint = np.delete(traint, np.arange(fold_indices[fold], fold_indices[fold+1]), 0)

            # find the least-squares fit!
            # NOTE: YOU NEED TO FILL THIS IN
            w = "blah" #### YOUR CODE HERE, remove the blah ####

            # calculate and record the mean squared losses

            fold_pred = np.dot(foldX, w)  # model predictions on held-out fold
            cv_loss[fold, p] = np.mean(np.power(fold_pred - foldt, 2))

            ind_pred = np.dot(testX[:, 0:p+1], w)   # model predictions on independent test data
            ind_loss[fold, p] = np.mean(np.power(ind_pred - testt, 2))

            train_pred = np.dot(trainX, w)  # model predictions on training data
            train_loss[fold, p] = np.mean(np.power(train_pred - traint, 2))

    # The results look a little more dramatic if you display the loss
    # on a log scale, so the following scales the loss scores
    log_cv_loss = np.log(cv_loss)
    log_ind_loss = np.log(ind_loss)
    log_train_loss = np.log(train_loss)

    mean_log_cv_loss = np.mean(log_cv_loss, 0)
    mean_log_ind_loss = np.mean(log_ind_loss, 0)
    mean_log_train_loss = np.mean(log_train_loss, 0)

    print '\n----------------------\nResults for {0}'.format(title)
    print 'mean_log_train_loss:\n{0}'.format(mean_log_train_loss)
    print 'mean_log_cv_loss:\n{0}'.format(mean_log_cv_loss)
    print 'mean_log_ind_loss:\n{0}'.format(mean_log_ind_loss)

    min_mean_log_cv_loss = min(mean_log_cv_loss)
    # TODO: has to be better way to get the min index...
    best_poly = [i for i, j in enumerate(mean_log_cv_loss) if j == min_mean_log_cv_loss][0]

    print 'minimum mean_log_cv_loss of {0} for order {1}'.format(min_mean_log_cv_loss, best_poly)

    # Plot log scale loss results
    plot_cv_results(log_train_loss, log_cv_loss, log_ind_loss, log_scale_p=True)

    # Uncomment to plot direct-scale loss results
    # plot_cv_results(train_loss, cv_loss, ind_loss, log_scale_p=True)

    return best_poly, min_mean_log_cv_loss


# -------------------------------------------------------------------------


def run_demo():
    # Parameters for synthetic data model
    # t = x - x^2 + 5x^3 + N(0, sigma)
    w = np.array([0, 1, 5, 2])
    xmin = -6
    xmax = 6
    sigma = 50

    x, t = generate_synthetic_data(100, w, xmin=xmin, xmax=xmax, sigma=sigma)
    testx, testt, = generate_synthetic_data(1000, w, xmin=xmin, xmax=xmax, sigma=sigma)

    plot_synthetic_data(x, t, w)

    K = 10

    run_cv( K, 7, x, t, testx, testt, randomize_data=False, title='{0}-fold CV'.format(K) )


# -------------------------------------------------------------------------
# SCRIPT
# -------------------------------------------------------------------------


run_demo()

plt.show()
