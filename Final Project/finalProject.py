from Perceptron import *
from finalProjectUtilityFunctions import *
from sklearn.linear_model import Perceptron
import pandas
import itertools

trainingSet, testingSet = getDataSets()

frame = pandas.DataFrame(columns=['state', 'label', 'features'])

for index, value in enumerate(testingSet):
    frame.loc[index] = [value['state'], value['label'], value['features']]

features = pandas.Series(list(itertools.chain(*[t['features'] for t in trainingSet])))
features = features.value_counts()
print "Number of Features: %i" % features.size

# Feature Vectors
print "Building Features Vectors..."
X_train = makeFeaturesVectors([t['features'] for t in trainingSet], features.index)
X_test = makeFeaturesVectors([t['features'] for t in testingSet], features.index)
