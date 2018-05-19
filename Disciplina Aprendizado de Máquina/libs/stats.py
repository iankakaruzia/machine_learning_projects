import numpy as np

# Mean
def mean(x):
    return np.sum(x)/float(len(x))

# Standard Deviation
def stdev(x):
    return np.sqrt(np.sum((i - mean(x)) ** 2 for i in x) / len(x))

# Variance
def var(y):
    return stdev(y) ** 2