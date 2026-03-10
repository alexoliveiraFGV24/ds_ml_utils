import numpy as np
import matplotlib.pyplot as plt
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

def plot_polynomial_regression(X, y, degree):
    
    # Compute the polynomial regression coefficients
    w = polynomial_regression(X, y, degree)
    
    # Generate test points for a smooth curve
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    
    # Create a design matrix for the test points
    phi_test = np.zeros((X_test.shape[0], degree + 1))

    # Fill the test design matrix with polynomial terms
    for i in range(degree + 1):
        phi_test[:, i] = X_test.flatten() ** i

    # Compute predicted values using the learned coefficients
    y_pred = phi_test @ w  

    fig = plt.figure(figsize=(8, 6))

    # Plot the original data points
    plt.scatter(X, y, color='blue', label='Data')

    # Plot the polynomial regression curve
    plt.plot(X_test, y_pred, color='red', label=f'Polynomial Regression (degree {degree})')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Polynomial Regression (degree {degree})")
    
    plt.show()

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

def plot_polynomial_regression_sklearn(X: np.array, y: np.array, degree: int):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    # Training the model
    model.fit(X, y)

    # Making predictions
    X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)  # Smoothing the graph
    y_pred = model.predict(X_pred).reshape(-1, 1)

    # Plotting the regressions
    fig = plt.figure(figsize=(8, 6))

    plt.scatter(X, y, color="blue", label="data")
    plt.plot(X_pred, y_pred, color="red", label=f"Polynomial regression (degree {degree})")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Polynomial Regression with scikit-learn") 
    plt.show()
