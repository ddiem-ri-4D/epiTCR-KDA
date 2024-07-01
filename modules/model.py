import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, f1_score, roc_curve, auc
)
from sklearn.model_selection import cross_validate, GridSearchCV

def check_false_pos_neg(model_test, test, pX_test, py_test):
    y_true = py_test["binder"].to_numpy()
    
    predicted_probabilities = model_test.predict(pX_test)
    predicted_labels = (predicted_probabilities >= 0.5).astype(int)
    y_pred = predicted_labels

    FP_index = test.index[(y_pred == 1) & (y_pred != y_true)]
    FN_index = test.index[(y_pred == 0) & (y_pred != y_true)]

    false_positive = test.loc[FP_index, "epitope"].value_counts().rename("number of false positive").to_frame()
    false_negative = test.loc[FN_index, "epitope"].value_counts().rename("number of false negative").to_frame()

    total_pos = test[test["binder"] == 1]["epitope"].value_counts().rename("number of positive").to_frame()
    total_neg = test[test["binder"] == 0]["epitope"].value_counts().rename("number of negative").to_frame()
    total = test["epitope"].value_counts().rename("total in testset").to_frame()

    false_pos_neg_loc = pd.concat([false_positive, false_negative, total_pos, total_neg, total], axis=1).fillna(0)
    false_pos_neg_loc["epitope"] = false_pos_neg_loc.index

    return false_pos_neg_loc[[
        "epitope",
        "number of false positive",
        "number of false negative",
        "number of positive",
        "number of negative",
        "total in testset"
    ]]

def fn_lst_unseen(data_train, data_test):
    lst_pep_train = data_train.epitope.unique().tolist()
    lst_pep_test = data_test.epitope.unique().tolist()
    
    unseen_epitopes = [item for item in lst_pep_test if item not in lst_pep_train]
    return unseen_epitopes, len(unseen_epitopes)

def _rocAuc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.legend(loc=4)
    plt.show()

def confusionMatrix(y_true, y_pred):
    target_names = ['Non-binder', 'Binder']
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.show()

def data_visu(data):
    y_test = data["binder"].to_numpy()
    y_test_pred = data["binder_pred"].to_numpy()

    confusionMatrix(y_test, y_test_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    accuracy = accuracy_score(y_test, y_test_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc_value = roc_auc_score(y_test, data["proba_pred"])

    print(f"AUC: {auc_value:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Sensitivity (TPR): {sensitivity:.3f}")
    print(f"Specificity (TNR): {specificity:.3f}")

    _rocAuc(y_test, data["proba_pred"])
