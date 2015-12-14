from Perceptron import *
from sklearn.linear_model import Perceptron
import random
import math


def getDataSets():
    # open input file
    data = open('foodTweets.txt', 'r')
    allData = data.readlines()

    # shuffle data so get different sets each time
    random.shuffle(allData)

    # train on 80% of data, test on 20%
    trainingSize = int(math.ceil(len(allData) * 0.8))
    trainingSet = allData[:trainingSize]
    testingSet = allData[trainingSize:]

    return trainingSet, testingSet
