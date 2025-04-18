from other_funcs.sigmoid import sigmoid
from other_funcs.error_funcs import error_logistic
from other_funcs.normalize_data import normalize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def logistic_regression(X_train, target, learning_rate=1e-8, epochs=1000, mode='batch'):
    m,n = X_train.shape
    error_hist = []

    phi_matrix = normalize(X_train)
    target = target.reshape(-1, 1)
    phi_matrix = np.hstack((np.ones((m, 1)), phi_matrix))

    w = np.random.normal(0, 0.01, (n+1, 1))
    w = w.reshape(-1, 1)
    
    if mode == 'sgd':
        for epoch in range(epochs):
            total_error = 0
            for i in range(m):
                xi = phi_matrix[i].reshape(1, -1) 
                yi = target[i]                     
                y_pred = sigmoid(xi @ w)           

                error = error_logistic(y_pred, yi)
                total_error += error

                gradient_logistic = (y_pred - yi) * xi.T 
                w -= learning_rate * gradient_logistic

            error_hist.append(total_error / m)  # Pode fazer a média do erro total
            # Pode fazer também o erro em cada iteração também (acho mais razoável)
            
    else:
        for epoch in range(epochs):
            y = sigmoid(phi_matrix @ w)
            y = y.reshape(-1, 1)
            error = error_logistic(y, target) / len(y)
            error_hist.append(error)
            gradient_logistic = phi_matrix.T @ (y - target)
            w = w - learning_rate * gradient_logistic
        
    return phi_matrix, w, error_hist

def get_predicted_class(y_proba):
    m, _ = y_proba.shape
    predicted_class = []
    
    for i in range(m):
        point_class = np.argmax(y_proba[i])
        predicted_class.append(point_class)
        
    return predicted_class

def get_accuracy(predicted_values, true_labels):
    
    n = len(true_labels)
    correct_pred = 0
    total_values = n
    for i in range(n):
        if predicted_values[i] == true_labels[i]:
            correct_pred += 1
    
    # Calculating the fpr and the tpr
    accuracy = correct_pred / total_values

    return accuracy

def logistic_regression_sklearn(X_train, y_train, X_test, y_test):
    model = LogisticRegression(multi_class='multinomial',solver='newton-cg')
    model.fit(X_train, y_train)
    y_pred_sm = model.predict_proba(X_test)
    y_pred = get_predicted_class(y_pred_sm)
    accuracy = model.score(X_test, y_test)
    
    return y_pred, accuracy

def plot_sse(error_history, title="Error Evolution History (SGD - Logistic)", epochs=1000):
    
    fig = plt.figure(figsize=(7, 5.5))
    plt.plot(error_history, color='blue')
    plt.xlabel(f"Iterations ({epochs} epochs)")
    plt.ylabel("Error")
    plt.title(title)
    plt.legend(["Error"])
    plt.show()
