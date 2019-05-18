from numpy import array
from kNN import classify0


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    test = array([0.0, 0.0])
    return group, labels, test

def createDataSet1():
    group = array([[1.0, 1.1, 1.2], [1.0, 1.0, 1.3], [0, 0, 1.5], [0, 0.1, 1.6]])
    labels = ['A', 'A', 'B', 'B']
    test = array([0.0, 0.0, 1.1])
    return group, labels, test


data_set, labels, test = createDataSet()
result = classify0(test, data_set, labels, 3)
print test, result

data_set, labels, test = createDataSet1()
result = classify0(test, data_set, labels, 3)
print test, result

