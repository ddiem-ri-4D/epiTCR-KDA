import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)

def analyze_false_predictions(test_data, y_true, y_pred):
    false_positives = test_data[(y_pred == 1) & (y_true != y_pred)]
    false_negatives = test_data[(y_pred == 0) & (y_true != y_pred)]

    fp_counts = false_positives['epitope'].value_counts().rename("number of false positives")
    fn_counts = false_negatives['epitope'].value_counts().rename("number of false negatives")
    positive_counts = test_data[test_data["binder"] == 1]['epitope'].value_counts().rename("number of positives")
    negative_counts = test_data[test_data["binder"] == 0]['epitope'].value_counts().rename("number of negatives")
    total_counts = test_data['epitope'].value_counts().rename("total in test set")

    summary_df = pd.concat([fp_counts, fn_counts, positive_counts, negative_counts, total_counts], axis=1).fillna(0)
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={'index': 'epitope'}, inplace=True)
    
    return summary_df

def list_unseen_epitopes(train_data, test_data):
    train_epitopes = set(train_data.epitope.unique())
    test_epitopes = set(test_data.epitope.unique())
    
    unseen_epitopes = test_epitopes - train_epitopes
    return list(unseen_epitopes), len(unseen_epitopes)

def plot_roc_auc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def display_confusion_matrix(y_true, y_pred):
    labels = ['Non-binder', 'Binder']
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()
    print(classification_report(y_true, y_pred, target_names=labels))

def visualize_data_performance(data):
    y_true = data["binder"].to_numpy()
    y_pred = data["binder_pred"].to_numpy()
    proba_pred = data["proba_pred"].to_numpy()

    display_confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc_value = roc_auc_score(y_true, proba_pred)

    print(f"AUC: {auc_value:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Sensitivity (Recall): {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")

    plot_roc_auc(y_true, proba_pred)
