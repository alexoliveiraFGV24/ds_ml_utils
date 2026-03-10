import numpy as np

def normalize(X:np.array) -> np.array:
    X_copy = X.copy()
    return (X_copy - np.mean(X_copy, axis=0)) / np.std(X_copy, axis=0)
