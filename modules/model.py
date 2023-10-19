import sklearn.metrics as metrics

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,classification_report,roc_curve,auc, f1_score
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import functools

def check_false_pos_neg(model_test, test, pX_test, py_test):
    y_true = py_test["binder"]
    
    predicted_probabilities = model_test.predict(pX_test)
    predicted_labels = predicted_probabilities.argmax(axis=1)
    predicted_labels = (predicted_probabilities >= 0.5).astype(int)
    y_pred = predicted_labels

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
    
    # Rename the columns with duplicate 'count' labels
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

    return false_pos_neg_loc


def modeling_kd(data_test, lst_unseen):
    FILTER_METRIC = data_test.copy()
    FILTER_METRIC.to_csv("./PREDICTION/KD_PREDICTION_Universal_v2_Overral.csv", index=False)
    lst_unseen = lst_unseen
    lst_dominant = ["GLCTLVAML", "NLVPMVATV", "GILGFVFTL", "TPRVTGGGAM", "ELAGIGILTV", "AVFDRKSDAK", "KLGGALQAK"]
    
    seen_data = FILTER_METRIC[~FILTER_METRIC.epitope.isin(lst_unseen)]
    seen_data.to_csv("./PREDICTION/KD_PREDICTION_Universal_v2_Seen.csv", index=False)
    unseen_data = FILTER_METRIC[FILTER_METRIC.epitope.isin(lst_unseen)]
    unseen_data.to_csv("./PREDICTION/KD_PREDICTION_Universal_v2_Unseen.csv", index=False)
    
    dominant_data = FILTER_METRIC[FILTER_METRIC.epitope.isin(lst_dominant)]
    dominant_data.to_csv("./PREDICTION/KD_PREDICTION_Universal_v2_Dominant.csv", index=False)
    nondominant_data = FILTER_METRIC[~FILTER_METRIC.epitope.isin(lst_dominant)]
    nondominant_data.to_csv("./PREDICTION/KD_PREDICTION_Universal_v2_NoDominant.csv", index=False)
    
    data_visu(FILTER_METRIC)
    data_visu(seen_data)
    data_visu(unseen_data)
    data_visu(dominant_data)
    data_visu(nondominant_data)
    
def fn_lst_unseen(data_train, data_test):
    lst_pep_train = data_train.epitope.unique().tolist()
    lst_pep_test = data_test.epitope.unique().tolist()
    
    res = [item for item in lst_pep_test if item not in lst_pep_train]
    return res, len(res)


def _rocAuc(y_true, y_score):
    y_pred01_proba = y_score.to_numpy()
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred01_proba)
    auc = metrics.roc_auc_score(y_true, y_pred01_proba)
    plt.plot(fpr,tpr,label="AUC = "+str(auc))
    plt.legend(loc=4)
    plt.show()

def confusionMatrix(y_true, y_pred):
    target_names = ['Non-binder', 'Binder']
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=target_names, columns=target_names)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted values')
    plt.ylabel('Actual values')
    plt.show()

def data_visu(data):
    y_test = data["binder"].to_numpy()
    y_test_pred = data["binder_pred"].to_numpy()

    confusionMatrix(y_test, y_test_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    accuracy = float(accuracy_score(y_test, y_test_pred).ravel())
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    auc = metrics.roc_auc_score(y_test, data["proba_pred"])
    print ("AUC : ", round(auc, 3))
    print ("Accuracy score  : ", round(accuracy, 3))
    print('Sensitivity (TPR): ', round(sensitivity, 3))
    print('Specificity (TNR): ', round(specificity, 3))

    _rocAuc(y_test, data["proba_pred"])
    
    
