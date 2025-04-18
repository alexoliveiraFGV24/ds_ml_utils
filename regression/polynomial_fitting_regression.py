import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def polynomial_regression(X: np.array, y: np.array, degree: int) -> tuple:
    """Perform polynomial regression on input data.
    
    Parameters
    ----------
    X : np.array
        Input feature matrix (n_samples, n_features)
    y : np.array
        Target values (n_samples,)
    degree : int
        Degree of the polynomial features
    
    Returns
    -------
    tuple
        (w, phi_matrix) where:
        - w : np.array
            Learned weights of the polynomial regression
        - phi_matrix : np.array
            Transformed feature matrix with polynomial features
    
    Notes
    -----
    Supports both univariate and multivariate input data. For multivariate data,
    creates polynomial features up to specified degree for each input feature.
    """
    y = y.reshape(-1, 1)
    rows = X.shape[0]

    if X.shape[1] == 1:  # One feature
        X = X.reshape(-1, 1)
        cols = degree + 1
        phi_matrix = np.zeros((rows, cols))
        for i in range(cols):
            phi_matrix[:, i] = X.flatten() ** i 
            
    elif X.shape[1] > 1:  # Multiple features
        cols = X.shape[1] * degree
        phi_matrix = np.zeros((rows, cols))
        for j in range(X.shape[1]):
            for i in range(degree):
                phi_matrix[:, i + j*degree] = X[:, j] ** (i+1)
        phi_matrix = np.hstack([np.ones((phi_matrix.shape[0], 1)), phi_matrix])  # Bias

    phi_matrix_pinv = np.linalg.inv(phi_matrix.T @ phi_matrix) @ phi_matrix.T
    w = phi_matrix_pinv @ y
    
    w = w.reshape(-1,1)
    return w, phi_matrix

def polynomial_regression_sklearn(X: np.array, y: np.array, degree: int) -> tuple:
    """Perform polynomial regression using sklearn's PolynomialFeatures and LinearRegression.
    
    Parameters
    ----------
    X : np.array
        Input feature matrix (n_samples, n_features)
    y : np.array
        Target values (n_samples,)
    degree : int
        Degree of the polynomial features
    
    Returns
    -------
    tuple
        (w, phi_matrix) where:
        - w : np.array
            Learned weights of the polynomial regression
        - phi_matrix : np.array
            Transformed feature matrix with polynomial features
    
    Notes
    -----
    This function uses sklearn's PolynomialFeatures to create polynomial features.
    """
    poly = PolynomialFeatures(degree)
    phi_matrix = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(phi_matrix, y)
    
    w = model.coef_.reshape(-1, 1)
    
    return w, phi_matrix
