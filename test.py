import random
import glob
import itertools
import warnings
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import functools

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
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,roc_auc_score,classification_report, roc_curve, auc,f1_score
from sklearn.metrics import classification_report

import modules.architectures as KD
import modules.processor as Processor
import modules.model as Model

warnings.filterwarnings('ignore')

from argparse import ArgumentParser

#Args parse
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-te", "--testfile", help="Specify the full path of the file with TCR sequences")
parser.add_argument("-sm", "--savedmodel", help="Specify saved model file")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="Specify output file")

args = parser.parse_args()

# print('Loading and encoding the dataset..')
print("###---LOADING DATA")

DATA_TEST = pd.read_parquet(args.testfile)
DATA_TEST = DATA_TEST[["CDR3b", "epitope"]

print("###---DATA REPRESENTATION")

X_TEST = Processor.DATA_REPRESENTATION(DATA_TEST)
X_TEST_cv = Processor.cv_data_kd(X_TEST)

# student_scratch.save("model.h5")
# student_scratch.save(args.savedmodel)
student_scratch = load_model(args.savedmodel)

print("###---EVALUATION")

predicted_probabilities = student_scratch.predict(X_TEST_cv)
predicted_labels = predicted_probabilities.argmax(axis=1)
predicted_labels = (predicted_probabilities >= 0.5).astype(int)

df_label = pd.DataFrame(zip(predicted_probabilities.squeeze(), predicted_labels.squeeze()), columns=['proba_pred', 'binder_pred'])
data_pred = pd.concat([DATA_TEST, df_label], axis=1)

print("###---SAVE DATA")

data_pred.to_parquet(args.outfile)
