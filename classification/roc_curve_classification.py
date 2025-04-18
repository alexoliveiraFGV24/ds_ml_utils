import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def get_fpr_tpr(predicted_values, true_labels, threshold):
    
    # Transforming in a nd array (if not)
    predicted_values = np.array(predicted_values)
    true_labels = np.array(true_labels)
    
    # Considering only the second column (as was explained in class) (prob de sucesso)
    predicted_values = predicted_values[:, 1]
    
    # Transforming the predicted values with the threshold
    # We'll consider >= threshold as 1 and < threshold as 0
    predicted_labels = (predicted_values >= threshold).astype(int)
    
    # Counting the tp, fp, tn, fn
    # We'll consider 1 == True and 0 == False
    tp = np.sum((predicted_labels == 1) & (true_labels == 1))
    fp = np.sum((predicted_labels == 1) & (true_labels == 0))
    tn = np.sum((predicted_labels == 0) & (true_labels == 0))
    fn = np.sum((predicted_labels == 0) & (true_labels == 1))
    
    # Calculating the fpr and the tpr
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
    return fpr, tpr

def plot_roc_curve(predicted_values, true_labels, thresholds, color="blue", label="roc"):

    """
    Plots the ROC curve for a given set of predicted values and true labels.
    
    Parameters:
    predicted_values (numpy.ndarray): The predicted probabilities for the positive class.
    true_labels (numpy.ndarray): The true binary labels (0 or 1).
    thresholds (list): A list of thresholds to evaluate.
    
    Returns:
    None: Displays the ROC curve plot.
    """
    
    # Initialize lists to store false positive rates and true positive rates
    roc_fpr, roc_tpr = set(), set()
    
    # Calculate FPR and TPR for each threshold
    for threshold in thresholds:
        fpr, tpr = get_fpr_tpr(predicted_values, true_labels, threshold)
        roc_fpr.add((fpr, tpr))
    
    roc_fpr, roc_tpr = zip(*sorted(roc_fpr))
    
    
    plt.figure(figsize=(8, 6))
    plt.plot(roc_fpr, roc_tpr, label=label, color=color)
    plt.plot([0,1], [0,1], linestyle="--", color="yellow", label="AUC=0.5")
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.legend()
    plt.grid()
    plt.show()

def get_auc(predicted_values, true_labels):
    """
    Computes the Area Under the Curve (AUC) for the ROC curve.
    
    Parameters:
    predicted_values (numpy.ndarray): The predicted probabilities for the positive class.
    true_labels (numpy.ndarray): The true binary labels (0 or 1).
    
    Returns:
    float: The AUC score.
    """
    
    # Transforming in a nd array (if not)
    predicted_values = np.array(predicted_values)
    true_labels = np.array(true_labels)
    
    # Considering only the second column (sucess probability)
    predicted_values = predicted_values[:, 1]
    
    return roc_auc_score(true_labels, predicted_values)
