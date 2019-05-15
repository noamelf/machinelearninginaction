from numpy import array
from kNN import classify0, createDataSet

group, labels = createDataSet()

inX = array([0.0, 0.0])

classify0(inX, group, labels, 3)
