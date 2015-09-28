__author__ = 'Kya'
import numpy as np
import matplotlib.pyplot as plt


def random_permutation_matrix(n):
    rows = np.random.permutation(n)
    cols = np.random.permutation(n)
    m = np.zeros((n, n))
    for r, c in zip(rows, cols):
        m[r][c] = 1
    return m


def permute_rows(X, P=None):
    if P is None:
        P = random_permutation_matrix(X.shape[0])
    return np.dot(P, X)


def permute_cols(X, P=None):
    if P is None:
        P = random_permutation_matrix(X.shape[0])
    return np.dot(X, P)


def read_data(filepath, d=','):
    return np.genfromtxt(filepath, delimiter=d, dtype=None)


def plot_data(x, t):
    plt.figure()
    plt.scatter(x, t, edgecolor='b', color='w', marker='o')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Data')
    plt.pause(.1)


def plot_model(x, w, color='r'):
    plotx = np.linspace(min(x) - 0.25, max(x) + 0.25, 100)
    plotX = np.zeros((plotx.shape[0], w.size))
    for k in range(w.size):
        plotX[:, k] = np.power(plotx, k)
    plott = np.dot(plotX, w)
    plt.plot(plotx, plott, color=color, linewidth=2)
    plt.pause(.1)
    return plotx, plott


def generate_synthetic_data(N, w, xmin=-5, xmax=5, sigma=150):
    x = (xmax - xmin) * np.random.rand(N) + xmin

    X = np.zeros((x.size, w.size))
    for k in range(w.size):
        X[:, k] = np.power(x, k)
    t = np.dot(X, w) + sigma * np.random.randn(x.shape[0])

    return x, t


def plot_synthetic_data(x, t, w, filepath=None):
    plot_data(x, t)
    plt.title('Plot of synthetic data; green curve is original generating function')
    plot_model(x, w, color='g')
    if filepath:
        plt.savefig(filepath, format='pdf')


def plot_cv_results(train_loss, cv_loss, log_scale_p=False):
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
    # ind_loss_mean = np.mean(ind_loss, 0)

    min_ylim = min(list(train_loss_mean) + list(cv_loss_mean))
    min_ylim = int(np.floor(min_ylim))
    max_ylim = max(list(train_loss_mean) + list(cv_loss_mean))
    max_ylim = int(np.ceil(max_ylim))

    plt.subplot(121)
    plt.plot(x, train_loss_mean, linewidth=2)
    plt.xlabel('Model Order')
    plt.ylabel(ylabel)
    plt.title('Train Loss')
    plt.pause(.1)
    plt.ylim(min_ylim, max_ylim)

    plt.subplot(122)
    plt.plot(x, cv_loss_mean, linewidth=2)
    plt.xlabel('Model Order')
    plt.ylabel(ylabel)
    plt.title('CV Loss')
    plt.pause(.1)
    plt.ylim(min_ylim, max_ylim)

    plt.subplots_adjust(right=0.95, wspace=0.4)
    plt.draw()


def run_cv(K, x, t, lambd, randomize_data=False, title='CV'):
    N = x.shape[0]
    p = 7

    if randomize_data:
        P = random_permutation_matrix(x.size)
        x = permute_rows(x, P)
        t = permute_rows(t, P)

    X = np.zeros((x.shape[0], p + 1))

    fold_indices = map(lambda x: int(x), np.linspace(0, N, K + 1))

    cv_loss = np.zeros((K))
    train_loss = np.zeros((K))

    X[:, p] = np.power(x, p)

    for fold in range(K):
        foldX = X[fold_indices[fold]:fold_indices[fold + 1], 0:p + 1]
        foldt = t[fold_indices[fold]:fold_indices[fold + 1]]

        trainX = np.copy(X[:, 0:p + 1])
        trainX = np.delete(trainX, np.arange(fold_indices[fold], fold_indices[fold + 1]), 0)

        traint = np.copy(t)
        traint = np.delete(traint, np.arange(fold_indices[fold], fold_indices[fold + 1]), 0)

        w = np.dot(np.dot(trainX.transpose(), trainX) + N * lambd * np.eye(trainX.shape[1]),
                   np.dot(trainX.transpose(), traint))

        fold_pred = np.dot(foldX, w)
        cv_loss[fold] = np.mean(np.power(fold_pred - foldt, 2))

        train_pred = np.dot(trainX, w)
        train_loss[fold] = np.mean(np.power(train_pred - traint, 2))

    '''
    for p in range(maxorder + 1):
        X[:, p] = np.power(x, p)

        testX[:, p] = np.power(testx, p)

        for fold in range(K):
            foldX = X[fold_indices[fold]:fold_indices[fold + 1], 0:p + 1]
            foldt = t[fold_indices[fold]:fold_indices[fold + 1]]

            trainX = np.copy(X[:, 0:p + 1])
            trainX = np.delete(trainX, np.arange(fold_indices[fold], fold_indices[fold + 1]), 0)

            traint = np.copy(t)
            traint = np.delete(traint, np.arange(fold_indices[fold], fold_indices[fold + 1]), 0)

            w = np.dot(np.linalg.inv(np.dot(trainX.transpose(), trainX)), (np.dot(trainX.transpose(), traint)))

            fold_pred = np.dot(foldX, w)
            cv_loss[fold, p] = np.mean(np.power(fold_pred - foldt, 2))

            ind_pred = np.dot(testX[:, 0:p + 1], w)
            ind_loss[fold, p] = np.mean(np.power(ind_pred - testt, 2))

            train_pred = np.dot(trainX, w)
            train_loss[fold, p] = np.mean(np.power(train_pred - traint, 2))
    '''
    log_cv_loss = np.log(cv_loss)
    # log_ind_loss = np.log(ind_loss)
    log_train_loss = np.log(train_loss)

    mean_log_cv_loss = np.mean(log_cv_loss, 0)
    # mean_log_ind_loss = np.mean(log_ind_loss, 0)
    mean_log_train_loss = np.mean(log_train_loss, 0)

    print '\n----------------------\nResults for {0}'.format(title)
    print 'mean_log_train_loss:\n{0}'.format(mean_log_train_loss)
    print 'mean_log_cv_loss:\n{0}'.format(mean_log_cv_loss)
    # print 'mean_log_ind_loss:\n{0}'.format(mean_log_ind_loss)

    # min_mean_log_cv_loss = min(mean_log_cv_loss)
    # TODO: has to be better way to get the min index...
    #best_poly = [i for i, j in enumerate(mean_log_cv_loss) if j == min_mean_log_cv_loss][0]

    # print 'minimum mean_log_cv_loss of {0} for order {1}'.format(min_mean_log_cv_loss, best_poly)
    print 'lambda value: %s; mean_log_cv_loss: %s' % (lambd, mean_log_cv_loss)

    #plot_cv_results(log_train_loss, log_cv_loss, log_scale_p=True)

    # Uncomment to plot direct-scale loss results
    # plot_cv_results(train_loss, cv_loss, ind_loss, log_scale_p=True)

    #return best_poly, min_mean_log_cv_loss


def run_problem():
    lambd = [0, 0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]

    data = read_data('synthdata2015.csv')
    x = data[:, 0]
    t = data[:, 1]

    K = 10

    for lam in lambd:
        run_cv(K, x, t, lam, randomize_data=True, title='regularized squares')


run_problem()

plt.show()
