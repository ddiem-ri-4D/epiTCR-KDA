import numpy as np
import pandas as pd
import warnings
import os
import sys
from keras.models import load_model
from argparse import ArgumentParser
import modules.processor as Processor

warnings.filterwarnings('ignore')

# Args parse
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-te", "--testfile", help="Specify the full path of the file with TCR sequences")
parser.add_argument("-sm", "--savedmodel", help="Specify saved model file")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="Specify output file")

args = parser.parse_args()

# Load and encode the dataset
print("###---LOADING DATA")
DATA_TEST = pd.read_parquet(args.testfile)

DATA_TEST = Processor.check_length_tcr(DATA_TEST)
DATA_TEST = Processor.check_length_epi(DATA_TEST)
DATA_TEST = DATA_TEST.reset_index(drop=True)

print("###---DATA REPRESENTATION")
X_TEST = Processor.DATA_REPRESENTATION(DATA_TEST)
X_TEST_cv = Processor.cv_data_kd(X_TEST)

# Load pre-trained model
print("###---LOAD MODEL PRETRAINING")
student_scratch = load_model(args.savedmodel)

# Evaluation
print("###---EVALUATION")
predicted_probabilities = student_scratch.predict(X_TEST_cv)
predicted_labels = (predicted_probabilities >= 0.5).astype(int)

df_label = pd.DataFrame({'proba_pred': predicted_probabilities.squeeze(), 'binder_pred': predicted_labels.squeeze()})
data_pred = pd.concat([DATA_TEST, df_label], axis=1)

# Save the predictions
print("###---SAVE DATA")
data_pred.to_parquet(args.outfile)
