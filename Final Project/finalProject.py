from Perceptron import Perceptron as MyPerceptron
from finalProjectUtilityFunctions import *
from sklearn.linear_model import Perceptron as SkPerceptron
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
trainingVectors = makeFeaturesVectors([t['features'] for t in trainingSet], features.index)
testingVectors = makeFeaturesVectors([t['features'] for t in testingSet], features.index)

# train my implementaion of the Perceptron
myPerceptron = MyPerceptron(numClasses=2, epochs=100, learningRate=1.5)
myPerceptron.train(trainingVectors, [t['label'] for t in trainingSet])

# get the predictions for my Perceptron
frame['myPerceptron'] = pandas.Series(
    [myPerceptron.predict(testingVectors[i, :]) for i in xrange(testingVectors.shape[0])])

# display results for class 0 - liberal
print '\nResults for the multinomial perceptron for class 0:\n'
printAccuracyPrecisionRecallF1(*computeAccuracyPrecisionRecallF1(
    *computeTrueFalsePostivesNegatives(frame['label'], frame['myPerceptron'], desiredClass=0)))

# display results for class 1 - conservative
print '\nResults for the multinomial perceptron for class 1:\n'
printAccuracyPrecisionRecallF1(*computeAccuracyPrecisionRecallF1(
    *computeTrueFalsePostivesNegatives(frame['label'], frame['myPerceptron'], desiredClass=1)))

# now do the same thing, but with the sklearn Perceptron
skPerceptron = SkPerceptron()
skPerceptron.fit(trainingVectors, [t['label'] for t in trainingSet])
frame['skPerceptron'] = skPerceptron.predict(testingVectors)

# display results for class 0 - liberal
print '\nResults for the multinomial perceptron for class 0:\n'
printAccuracyPrecisionRecallF1(*computeAccuracyPrecisionRecallF1(
    *computeTrueFalsePostivesNegatives(frame['label'], frame['skPerceptron'], desiredClass=0)))

# display results for class 1 - conservative
print '\nResults for the multinomial perceptron for class 1:\n'
printAccuracyPrecisionRecallF1(*computeAccuracyPrecisionRecallF1(
    *computeTrueFalsePostivesNegatives(frame['label'], frame['skPerceptron'], desiredClass=1)))
