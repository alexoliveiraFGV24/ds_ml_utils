import numpy as np
import matplotlib.pyplot as plt

def linear_regression_PI(X, y):
    """
    Implements a Linear Regression model using the Moore-Penrose pseudo-inverse
    
    Parameters
    ----------
    X : array
        A 2-dimensional array with samples in the rows and features in the columns
    y : array
        An array with the same number of  as samples in X, the values to predict
    
    Returns
    -------
    w : array
        Learnt parameters
            
    Notes
    -----
    The first column of X corresponds to the bias (`w_0`)
    """
    
    # Converting to numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()
    
    # Creating the phi matrix (just the normal data)
    phi_matrix = X.copy()
    
    # Adding a column of ones to X
    phi_matrix = np.hstack([np.ones((phi_matrix.shape[0], 1)), phi_matrix])
    
    # Calculating the pseudo-inverse
    phi_matrix_pinv = np.linalg.inv(phi_matrix.T @ phi_matrix) @ phi_matrix.T
    # phi_matrix_pinv = np.linalg.pinv(phi_matrix)

    # Creating the learned parameters
    w = phi_matrix_pinv @ y
    
    return w

def linear_regression_SGD(X, y, lr=1e-8, epochs=10):
    """
    Implements a Linear Regression model using Stochastic Gradient Descent
    
    Parameters
    ----------
    X : array
        A 2-dimensional array with samples in the rows and features in the columns
    y : array
        An array with the same number of  as samples in X, the values to predict
    lr : float
        Learning rate
    epochs : int
        number of epochs to use for the gradient descent
    
    Returns
    -------
    w : array
        Learnt parameters
    sse_history : list
        A list that contains the error of the model in every iteration
        
    Notes
    -----
    This function uses the gradient of the sum of squares function (Equations 3.12, and 3.23 in the Bishop book)
    """
    
    # Convert to numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()
    
    phi_matrix = X.copy()
    
    # Adding a column of ones to X
    phi_matrix = np.hstack([np.ones((phi_matrix.shape[0], 1)), phi_matrix])

    # Initialize weights with random values from a normal distribution (small variation)
    w_0 = np.random.normal(0, 0.01, phi_matrix.shape[1])
    w = w_0.copy()

    # Store SSE
    sse_history = []

    # Perform Stochastic Gradient Descent (SGD)
    for epoch in range(epochs):
        
        # Shuffle the data in each epoch
        indices = np.random.permutation(phi_matrix.shape[0])
        
        for i in indices:
            
            # Get current data point
            phi_xi = phi_matrix[i]

            # Compute error and the sse for the current data point
            error = y[i] - np.dot(w, phi_xi)
            sse = 0.5 * (error ** 2)
            sse_history.append(sse)  # Store SSE

            # Calculate the gradient
            gradient = -error * phi_xi

            # Update weights
            w = w - lr * gradient

    return w, sse_history

def plot_sse(sse_history, title="SSE Evolution History (SGD)", epochs=100):
    """
    Plots the Sum Squared Error (SSE) history
    
    Parameters
    ----------
    sse_history : array
        An array that contains the error of the model in every iteration
    title : str
        The title of the plot
    """
    
    plt.plot(sse_history)
    plt.xlabel(f"Iterations ({epochs} epochs)")
    plt.ylabel("SSE")
    plt.title(title)
    plt.legend(["SSE"])
    plt.show()