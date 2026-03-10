import numpy as np
from sklearn.model_selection import train_test_split

def custom_train_test_split(X, y, train_size=0.2, random_state=None):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    X (array-like): Features.
    y (array-like): Target variable.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int, optional): Random seed for reproducibility.

    Returns:
    X_train, X_test, y_train, y_test: Split datasets.
    """
    return train_test_split(X, y, train_size=train_size, random_state=random_state)

def custom_train_test_validation_split(X, y, train_size=0.2, validation_size=0.5, random_state=42):
    """
    Splits the dataset into training, validation and testing sets.
    
    Parameters:
    X (array-like): Features.
    y (array-like): Target variable.
    train_size (float): Proportion of the dataset to include in the training split.
    validation_size (float): Proportion of the dataset to include in the validation split.
    random_state (int, optional): Random seed for reproducibility.
    
    Returns:
    X_training, y_training, X_train, y_train, X_validation, y_validation, X_test, y_test: Split datasets.
    """

    # Splitting the data into training, validation and test.
    X_train, X_split, y_train, y_split = train_test_split(X, y, train_size=train_size, random_state=random_state, shuffle=True)
    X_validation, X_test, y_validation, y_test = train_test_split(X_split, y_split, train_size=validation_size, random_state=random_state, shuffle=True)

    # Concatenate the training and validation sets
    X_training = np.concatenate((X_train, X_validation), axis=0)
    y_training = np.concatenate((y_train, y_validation), axis=0)
    
    return X_training, y_training, X_train, y_train, X_validation, y_validation, X_test, y_test
