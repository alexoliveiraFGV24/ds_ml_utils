import numpy as np

def rmse(y1: np.array, y2: np.array) -> float:
    """Calculate the Root Mean Square Error between two arrays.
    
    Parameters
    ----------
    y1 : np.array
        First input array of observed values
    y2 : np.array
        Second input array of predicted values
    
    Returns
    -------
    float
        The root mean square error between y1 and y2
    
    Examples
    --------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 1.9, 3.2])
    >>> rmse(y_true, y_pred)
    0.14142135623730953
    """
    return np.sqrt(np.mean((y1-y2)**2))

def mse(y1: np.array, y2: np.array) -> float:
    """Calculate the Mean Square Error between two arrays.
    
    Parameters
    ----------
    y1 : np.array
        First input array of observed values
    y2 : np.array
        Second input array of predicted values
    
    Returns
    -------
    float
        The mean square error between y1 and y2
    
    Examples
    --------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 1.9, 3.2])
    >>> mse(y_true, y_pred)
    0.020000000000000018
    """
    return np.mean((y1-y2)**2)

def sse(y1: np.array, y2: np.array) -> float:
    """Calculate the Sum of Squared Errors between two arrays.
    
    Parameters
    ----------
    y1 : np.array
        First input array of observed values
    y2 : np.array
        Second input array of predicted values
    
    Returns
    -------
    float
        The sum of squared errors between y1 and y2
    
    Examples
    --------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 1.9, 3.2])
    >>> sse(y_true, y_pred)
    0.020000000000000018
    """
    return np.sum((y1-y2)**2)