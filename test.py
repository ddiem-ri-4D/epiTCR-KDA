import numpy as np
import pandas as pd
import warnings
import os
import sys
import pandas as pd
import numpy as np
import modules.model as Model

from keras.models import load_model
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
DATA_TEST = pd.read_parquet(args.testfile)

DATA_TEST = Processor.check_length_tcr(DATA_TEST)
DATA_TEST = Processor.check_length_epi(DATA_TEST)
DATA_TEST = DATA_TEST.reset_index(drop=True)

###--DATA_REPRESENTATION
print("###---DATA REPRESENTATION")

X_TEST = Processor.DATA_REPRESENTATION(DATA_TEST)
X_TEST_cv = Processor.cv_data_kd(X_TEST)

###--LOAD MODEL PRETRAINING
student_scratch = load_model(args.savedmodel)

###---Evaluation
print("###---EVALUATION")

predicted_probabilities = student_scratch.predict(X_TEST_cv)
predicted_labels = predicted_probabilities.argmax(axis=1)
predicted_labels = (predicted_probabilities >= 0.5).astype(int)

df_label = pd.DataFrame(zip(predicted_probabilities.squeeze(), predicted_labels.squeeze()), columns=['proba_pred', 'binder_pred'])
data_pred = pd.concat([DATA_TEST, df_label], axis=1)

print("###---SAVE DATA")

data_pred.to_parquet(args.outfile)
