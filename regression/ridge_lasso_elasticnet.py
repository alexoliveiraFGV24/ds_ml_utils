from other_funcs.error_funcs import rmse
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet

def ridge_regression(X: np.array, y: np.array, penalty: float) -> tuple:
    
    # Creating the phi_matrix
    phi_matrix = X.copy()
    phi_matrix = np.hstack([np.ones((phi_matrix.shape[0], 1)), phi_matrix])
    identity = np.identity(phi_matrix.shape[1])
    
    # Learning the w and calculating the rmse error
    w = np.linalg.inv(penalty * identity + phi_matrix.T @ phi_matrix) @ phi_matrix.T @ y
    rmse_error = rmse(y, phi_matrix @ w)
    
    return w, rmse_error

def best_lambda(X_train, X_pred, y_train, y_pred, penalties):
    
    # Initializing the training and prediction rmse histories
    rmse_hist_training = []
    rmse_hist_pred = []

    # Creating the phi matrix for the prediction set
    phi_matrix_pred = X_pred.copy()
    phi_matrix_pred = np.hstack([np.ones((phi_matrix_pred.shape[0], 1)), phi_matrix_pred])

    # Finding the best penalty for ridge regression
    for penalty in penalties:
        w_train, rmse_train = ridge_regression(X_train, y_train, penalty)
        rmse_hist_training.append(rmse_train)
        
        # Calculating the error
        rmse_pred = rmse(y_pred, phi_matrix_pred @ w_train)
        rmse_hist_pred.append(rmse_pred)

    # Defining the best lambda
    best_lambda = penalties[np.argmin(rmse_hist_pred)]

    return best_lambda, rmse_hist_training, rmse_hist_pred

def lasso_regression(X, X_pred, y, y_pred, alpha):
    
    lasso_regression = Lasso(alpha=alpha).fit(X, y)
    y_predict_lasso_reg = lasso_regression.predict(X_pred)
    rmse_lasso = float(np.sqrt(np.mean((y_pred - y_predict_lasso_reg)**2)))  # Não sei pq não dá certo com a função que criei
    
    return y_predict_lasso_reg, rmse_lasso

def lasso_best_alpha(X, X_pred, y, y_pred, alpha_values):
    
    # Initializing the histories
    rmse_hist_x = []
    rmse_hist_pred = []

    # Learning the best alpha
    for alpha_value in alpha_values:
        _, rmse_x = lasso_regression(X, X, y, y, alpha_value)  # Aprendo os parâmetros (w) com o de treino
        _, rmse_pred = lasso_regression(X, X_pred, y, y_pred, alpha_value)  # Regulo o hiperparâmetro (alpha) com o de validação
        
        rmse_hist_x.append(rmse_x)
        rmse_hist_pred.append(rmse_pred)
    
    # Defining the best alpha
    best_alpha = alpha_values[np.argmin(rmse_hist_pred)]
    
    return best_alpha

def ElasticNet_Regression(X_training, X_testing, y_training, y_testing, alpha_value, lambda_value):

    # Creating the model for training 
    ELRegression = ElasticNet(alpha=alpha_value, l1_ratio=lambda_value).fit(X_training, y_training)
    y_predict_EL = ELRegression.predict(X_testing)

    # Calculating the rmse error
    rmse_EL = float(np.sqrt(np.mean((y_testing - y_predict_EL)**2)))
    
    return rmse_EL
