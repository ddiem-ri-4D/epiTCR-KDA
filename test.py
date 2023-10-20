import numpy as np
import pandas as pd
import random
import glob
import itertools
import warnings
import os
import sys
import pandas as pd
import numpy as np
import modules.model as Model
from imblearn.under_sampling import RandomUnderSampler

import sklearn.metrics as metrics
from keras.models import load_model

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,roc_auc_score,classification_report, roc_curve, auc,f1_score
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import functools

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
# parser.add_argument("-sm", "--savemodel", help="Specify the full path of the file with save model")

args = parser.parse_args()

# print('Loading and encoding the dataset..')
print("###---LOADING DATA")

DATA_TEST = pd.read_csv(args.testfile)
DATA_TEST = DATA_TEST[["CDR3b", "epitope", "binder"]]

# DATA_TEST = pd.read_csv("./DATA_FINAL/DATA_TEST_v2.csv")

###--DATA_REPRESENTATION
print("###---DATA REPRESENTATION")

X_TEST, y_TEST = Processor.DATA_REPRESENTATION(DATA_TEST),  DATA_TEST[["binder"]]
X_TEST_cv, y_TEST_cv = Processor.cv_data_kd(X_TEST), np.squeeze(np.array(y_TEST))


###--TRAINING


# student_scratch.save("model.h5")
# student_scratch.save(args.savedmodel)
student_scratch = load_model(args.savedmodel)

###---Evaluation
print("###---EVALUATION")

predicted_probabilities = student_scratch.predict(X_TEST_cv)
predicted_labels = predicted_probabilities.argmax(axis=1)
predicted_labels = (predicted_probabilities >= 0.5).astype(int)

df_label = pd.DataFrame(zip(predicted_probabilities.squeeze(), predicted_labels.squeeze()), columns=['proba_pred', 'binder_pred'])
data_pred = pd.concat([DATA_TEST, df_label], axis=1)

print("###---SAVE DATA")

data_pred.to_csv(args.outfile, index=False)

# lst_unseen_result, l_lst_unseen = Processor.fn_lst_unseen(DATA_TRAIN, DATA_TEST)
# Model.modeling_kd(data_pred,lst_unseen_result)

