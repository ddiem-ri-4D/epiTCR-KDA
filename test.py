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
test_data = pd.read_parquet(args.testfile)

test_data = Processor.check_length_tcr(test_data)
test_data = Processor.check_length_epi(test_data)
test_data = test_data.reset_index(drop=True)

# Data representation
print("###---DATA REPRESENTATION")
X_test = Processor.DATA_REPRESENTATION(test_data)
X_test_cv = Processor.cv_data_kd(X_test)

# Load pre-trained model
print("###---LOAD MODEL PRETRAINING")
loaded_model = load_model(args.savedmodel)

# Evaluation
print("###---EVALUATION")
predicted_probabilities = loaded_model.predict(X_test_cv)
predicted_labels = (predicted_probabilities >= 0.5).astype(int)

results_df = pd.DataFrame({'proba_pred': predicted_probabilities.squeeze(), 'binder_pred': predicted_labels.squeeze()})
predicted_data = pd.concat([test_data, results_df], axis=1)

# Save the predictions
print("###---SAVE DATA")
predicted_data.to_parquet(args.outfile)
