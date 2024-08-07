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

def check_false_pos_neg(test, y_true, y_pred):
    FP_index = []
    for idx in test.index:
        if y_pred[idx] == 1 and y_pred[idx] != y_true[idx]:
            FP_index.append(idx) 
    false_positive = pd.DataFrame(test.loc[FP_index, "epitope"].value_counts())
    false_positive.rename(columns={"epitope": "number of false positive"}, inplace = True)

    FN_index = []
    for idx in test.index:
        if y_pred[idx] == 0 and y_pred[idx] != y_true[idx]:
            FN_index.append(idx) 
    false_negative = pd.DataFrame(test.loc[FN_index, "epitope"].value_counts())
    false_negative.rename(columns={"epitope": "number of false negative"}, inplace = True)

    total_pos = pd.DataFrame(test[test["binder"]==1]["epitope"].value_counts())
    total_pos.rename(columns={"epitope": "number of positive"}, inplace = True)
    total_neg = pd.DataFrame(test[test["binder"]==0]["epitope"].value_counts())
    total_neg.rename(columns={"epitope": "number of negative"}, inplace = True)

    total = pd.DataFrame(test["epitope"].value_counts())
    total.rename(columns={"epitope": "total in testset"}, inplace = True)

    false_pos_neg_loc = pd.concat([false_positive, false_negative, total_pos, total_neg, total], axis=1).fillna(0)
    false_pos_neg_loc["epitope"] = false_pos_neg_loc.index
   # Get the column names in the DataFrame
    columns = false_pos_neg_loc.columns.tolist()
    
    # Rename the columns with duplicate 'count' binders
    false_pos_neg_loc.columns = [
        ("number of false positive" if (col == 'count' and idx == 0) else col) for idx, col in enumerate(false_pos_neg_loc.columns)
    ]
    false_pos_neg_loc.columns = [
        ("number of false negative" if (col == 'count' and idx == 1) else col) for idx, col in enumerate(false_pos_neg_loc.columns)
    ]
    false_pos_neg_loc.columns = [
        ("number of positive" if (col == 'count' and idx == 2) else col) for idx, col in enumerate(false_pos_neg_loc.columns)
    ]
    false_pos_neg_loc.columns = [
        ("number of negative" if (col == 'count' and idx == 3) else col) for idx, col in enumerate(false_pos_neg_loc.columns)
    ]
    false_pos_neg_loc.columns = [
        ("total in testset" if (col == 'count' and idx == 4) else col) for idx, col in enumerate(false_pos_neg_loc.columns)
    ]

    false_pos_neg_loc = false_pos_neg_loc[[
        "epitope",
        "number of false positive",
        "number of false negative",
        "number of positive",
        "number of negative",
        "total in testset"
    ]]

    return false_pos_neg_loc.reset_index(drop=True)

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
