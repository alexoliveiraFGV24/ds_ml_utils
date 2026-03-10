import numpy as np

def euclidean_dist(x1: np.array, x2: np.array) -> float:
    """Calculate Euclidean distance between two points.
    
    Parameters
    ----------
    x1 : np.array
        First point coordinates
    x2 : np.array
        Second point coordinates
    
    Returns
    -------
    float
        Euclidean distance between x1 and x2
    
    Examples
    --------
    >>> x1 = np.array([1, 2])
    >>> x2 = np.array([4, 6])
    >>> euclidean_dist(x1, x2)
    5.0
    """
    return np.linalg.norm(x1-x2)

def knn_predict_binary(X_train: np.array, y_train: np.array, point: np.array, k: int) -> int:
    """Predict binary class using k-Nearest Neighbors algorithm.
    
    Parameters
    ----------
    X_train : np.array
        Training data features (n_samples, n_features)
    y_train : np.array
        Training data labels (n_samples,)
    point : np.array
        Test point to classify (n_features,)
    k : int
        Number of nearest neighbors to consider
    
    Returns
    -------
    int
        Predicted class (0 or 1) based on majority vote
    
    Notes
    -----
    Handles cases where a training point is identical to the test point by
    excluding zero-distance points from the k nearest neighbors.
    """
    distances = []
    for i in range(len(X_train)):
        distance = euclidean_dist(point, X_train[i])
        true_label = y_train[i]
        distances.append((distance, true_label))
    
    distances.sort(key=lambda x: x[0])
    
    if distances[0][0] == 0:
        distances = distances[1:]

    k_nearest = np.array([label for distance, label in distances[:k]])
    count_0 = np.sum(k_nearest == 0)
    count_1 = np.sum(k_nearest == 1)
    
    return 0 if count_0 > count_1 else 1

def calculate_f1_score(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array, k: int) -> float:
    """Calculate F1 score for binary classification using KNN.
    
    Parameters
    ----------
    X_train : np.array
        Training data features (n_train_samples, n_features)
    X_test : np.array
        Test data features (n_test_samples, n_features)
    y_train : np.array
        Training data labels (n_train_samples,)
    y_test : np.array
        True test labels (n_test_samples,)
    k : int
        Number of nearest neighbors for KNN prediction
    
    Returns
    -------
    float
        F1 score (between 0 and 1), returns 0 if denominator is zero
    
    Notes
    -----
    F1 score is the harmonic mean of precision and recall.
    Formula: F1 = TP / (TP + 0.5 * (FP + FN))
    """
    predicted_labels = [knn_predict_binary(X_train, y_train, point, k) for point in X_test]
    predicted_labels = np.array(predicted_labels)
    
    tp = np.sum((predicted_labels == 1) & (y_test == 1))
    fp = np.sum((predicted_labels == 1) & (y_test == 0))
    fn = np.sum((predicted_labels == 0) & (y_test == 1))
    
    denominator = tp + 0.5 * (fp + fn)
    return tp / denominator if denominator > 0 else 0.0 
