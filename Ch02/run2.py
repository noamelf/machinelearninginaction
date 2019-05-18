#%%
from numpy import array
from machinelearninginaction.Ch02 import kNN

#%%
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat, datingLabels = kNN.file2matrix(
    '/home/noame/bv/ml-in-action/machinelearninginaction/Ch02/datingTestSet.txt'
)
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
           15.0 * array(datingLabels), 15.0 * array(datingLabels))
# ax.axis([-2, 25, -0.2, 2.0])
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')
plt.show()

#%%
from numpy import zeros,shape,tile
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  #element wise divide
    return normDataSet, ranges, minVals

#%%
dataSet = datingDataMat

minVals = dataSet.min(0)
maxVals = dataSet.max(0)
ranges = maxVals - minVals
normDataSet = zeros(shape(dataSet))
m = dataSet.shape[0]
normDataSet = dataSet - tile(minVals, (m, 1))
normDataSet = normDataSet / tile(ranges, (m, 1))  #element wise divide
# return normDataSet, ranges, minVals


#%%
minVals

#%%
m

#%%
tile(minVals, (m, 1))

#%%
ranges

#%%
normDataSet = dataSet - tile(minVals, (m, 1))
normDataSet / tile(ranges, (m, 1))

#%%
dataSet

#%%
hoRatio = 0.10  #hold out 10%
datingDataMat, datingLabels = file2matrix(
    '/home/noame/bv/ml-in-action/machinelearninginaction/Ch02/datingTestSet2.txt')  #load data setfrom file
normMat, ranges, minVals = autoNorm(datingDataMat)

#%%
m = normMat.shape[0]
numTestVecs = int(m * hoRatio)
errorCount = 0.0
for i in range(numTestVecs):
    classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                    datingLabels[numTestVecs:m], 3)
    print "the classifier came back with: %d, the real answer is: %d" % (
        classifierResult, datingLabels[i])
    if (classifierResult != datingLabels[i]): errorCount += 1.0
print "the total error rate is: %f" % (errorCount / float(numTestVecs))
print errorCount