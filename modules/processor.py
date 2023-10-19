import pandas as pd
import numpy as np
import pickle
import time
import sys
import os
from keras.models import load_model

import sklearn.metrics as metrics

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,roc_auc_score,classification_report,roc_curve,auc, f1_score
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import functools


def DATA_REPRESENTATION(DATA):
    DATA_cdr3 = getProteinByDiheral(DATA.CDR3b.unique(), "/home/jovyan/work/data01/work/Diem/SM07/TCR-ML/PHASE2_MODEL/DIHEDRAL/MODEL/DATA/PARSE_DIH_CDR3/")
    DATA_pep = getProteinByDiheral(DATA.epitope.unique(), "/home/jovyan/work/data01/work/Diem/SM07/TCR-ML/PHASE2_MODEL/DIHEDRAL/MODEL/DATA/PARSE_DIH_PEP/")

    DATA_TCRpep = DAtoDataFrame(DATA, DATA_cdr3, DATA_pep)

    DATA_TCRpep_SPLIT = DATA_TCRpep[['T1', 'T2',
           'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13',
           'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23',
           'T24', 'T25', 'T26', 'T27', 'T28', 'T29', 'T30', 'T31', 'T32', 'T33',
           'T34', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10',
           'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18']]
    
    return DATA_TCRpep_SPLIT


def getProteinByDiheral(list_seq, link):
    folder_path = link
    dict_lst = dict.fromkeys(list_seq)
    new_dict = {}

    for key, _ in dict_lst.items():
        csv_file = os.path.join(folder_path, key + ".tsv")
        if os.path.isfile(csv_file):
            df = pd.read_csv(csv_file, delimiter='\t', header=None)
            df.columns =['residueID', 'X_phi', 'Y_psi', 'label']
            df = df[["X_phi", "Y_psi"]]
            values = df.values.flatten().tolist()
            dict_lst[key] = values
    
    return dict_lst

def fn_downsampling(data):
    X_train, y_train = data[['T1', 'T2', 'T3',
       'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14',
       'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T24',
       'T25', 'T26', 'T27', 'T28', 'T29', 'T30', 'T31', 'T32', 'T33', 'T34',
       'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11',
       'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18']], data[["binder"]]

    nm = RandomUnderSampler(random_state=42)
    X_res, y_res = nm.fit_resample(X_train, y_train)
    X_res, y_res =  X_res.reset_index(drop=True), y_res.reset_index(drop=True)
    
    return X_res, y_res

def cvVectorDict2DFTCR(vector, num_columns=34):
    # Loại bỏ giá trị None khỏi vector
    vector = {key: value for key, value in vector.items() if value is not None}
    
    temp = pd.DataFrame.from_dict(vector, orient='index')
    temp.columns = ["T" + str(i) for i in range(1, temp.shape[1] + 1)]
    
    existing_columns = temp.columns
    for i in range(len(existing_columns) + 1, num_columns + 1):
        temp["T" + str(i)] = 0
    return temp

def cvVectorDict2DFEpitope(vector, num_columns=18):
    vector = {key: value for key, value in vector.items() if value is not None}
    
    temp = pd.DataFrame.from_dict(vector, orient='index')
    temp.columns = ["E" + str(i) for i in range(1, temp.shape[1] + 1)]
    
    existing_columns = temp.columns
    for i in range(len(existing_columns) + 1, num_columns + 1):
        temp["E" + str(i)] = 0
    return temp

def DAtoDataFrame(SAMPLE, SAMPLE_CDR3, SAMPLE_PEP):
    SAMPLE_CDR3 = cvVectorDict2DFTCR(SAMPLE_CDR3, 34)
    SAMPLE_PEP = cvVectorDict2DFEpitope(SAMPLE_PEP, 18)
    
    SAMPLE_CDR3_df = SAMPLE_CDR3.reset_index(drop=False).rename(columns={"index": "CDR3b"})
    SAMPLE_PEP_df = SAMPLE_PEP.reset_index(drop=False).rename(columns={"index": "epitope"})

    SAMPLE_SPLIT_cdr3_merge = SAMPLE.merge(SAMPLE_CDR3_df, how='left', on='CDR3b')
    SAMPLE_SPLIT_pep_merge = SAMPLE_SPLIT_cdr3_merge.merge(SAMPLE_PEP_df, how='left', on='epitope')
    
    # Fill missing values with 0 for columns T1 to T34
    columns_to_fill_T = ['T' + str(i) for i in range(1, 35)]
    SAMPLE_SPLIT_pep_merge[columns_to_fill_T] = SAMPLE_SPLIT_pep_merge[columns_to_fill_T].fillna(0)
    # print(SAMPLE_SPLIT_pep_merge[columns_to_fill_T][SAMPLE_SPLIT_pep_merge[columns_to_fill_T].T1.isna()])

    # Fill missing values with 0 for columns E1 to E18
    columns_to_fill_E = ['E' + str(i) for i in range(1, 19)]
    SAMPLE_SPLIT_pep_merge[columns_to_fill_E] = SAMPLE_SPLIT_pep_merge[columns_to_fill_E].fillna(0)
    # print(SAMPLE_SPLIT_pep_merge[columns_to_fill_E][SAMPLE_SPLIT_pep_merge[columns_to_fill_E].E1.isna()])
    
    SAMPLE_SPLIT_TCRpep = SAMPLE_SPLIT_pep_merge.dropna(subset=['T1', 'E1']).copy()
    
    return SAMPLE_SPLIT_TCRpep

def cv_data_kd(PTEST_X):
    PTEST_X_CDR3 = np.array(PTEST_X.iloc[:, :34].values.tolist())
    PTEST_X_CDR3_new = PTEST_X_CDR3.reshape((len(PTEST_X_CDR3), 17, 2))

    PTEST_X_epitope = np.array(PTEST_X.iloc[:, 34:].values.tolist())
    PTEST_X_epitope_new = PTEST_X_epitope.reshape((len(PTEST_X_epitope), 9, 2))    
    
    PTEST_X_cv = []
    nan_array = np.full((PTEST_X_CDR3_new.shape[1] - PTEST_X_epitope_new.shape[1], PTEST_X_epitope_new.shape[2]), 0)
    for i in range(len(PTEST_X_CDR3_new)):
        tmp = np.concatenate((PTEST_X_CDR3_new[i], \
                              np.concatenate((PTEST_X_epitope_new[i], nan_array), axis=0)), axis=1)
        PTEST_X_cv.append(tmp)

    PTEST_X_FULL_cv = np.expand_dims(np.array(PTEST_X_cv), axis=-1)
    return PTEST_X_FULL_cv

def fn_lst_unseen(data_train, data_test):
    lst_pep_train = data_train.epitope.unique().tolist()
    lst_pep_test = data_test.epitope.unique().tolist()
    
    res = [item for item in lst_pep_test if item not in lst_pep_train]
    return res, len(res)


