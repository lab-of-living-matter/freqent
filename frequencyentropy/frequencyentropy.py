import numpy as np


def corrMatrix(data):
    '''
    Takes the data matrix and returns a correlation matrix for every lag time

    Parameters
    ----------
    data : 2d numpy array
        data contains time series data of multiple variables. The first dimension
        contains each variable, and the second dimension contains the time series
        for that variable. e.g. data[0] returns time series for first variable.
    '''

    data = np.asarray(data)

    nrows, ncols = data.shape
    for x in range(1, 10):
        pass
