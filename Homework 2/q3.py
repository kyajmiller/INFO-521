def get_value(x):
    t = (-1.50718122e-02 * x) + 4.09241546e+01
    return t


def getSquaredError(predicted, actual):
    difference = actual - predicted
    squaredError = difference * difference
    return squaredError

prediction_2012 = get_value(2012)
print(prediction_2012)
squaredError_2012 = getSquaredError(prediction_2012, 10.75)
print(squaredError_2012)
