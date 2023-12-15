import numpy as np
import sys
import pandas as pd
import random
import glob
import itertools
import warnings
import os
import pandas as pd
import numpy as np
import modules.model as Model
from imblearn.under_sampling import RandomUnderSampler

import sklearn.metrics as metrics
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import load_model
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

import modules.architectures as KD
import modules.processor as Processor
import modules.model as Model

warnings.filterwarnings('ignore')

from argparse import ArgumentParser

#Args parse
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-tr", "--trainfile", help="Specify the full path of the training file with TCR sequences")
parser.add_argument("-te", "--testfile", help="Specify the full path of the file with TCR sequences")
parser.add_argument("-sm", "--savemodel", help="Specify save model file")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="Specify output file")

args = parser.parse_args()

# print('Loading and encoding the dataset..')

print("###---LOADING DATA")

DATA_TRAIN = pd.read_parquet(args.trainfile)
DATA_TEST = pd.read_parquet(args.testfile)

DATA_TRAIN = DATA_TRAIN[["CDR3b", "epitope", "binder"]]
DATA_TEST = DATA_TEST[["CDR3b", "epitope", "binder"]]

###--DATA_REPRESENTATION
print("###---DATA REPRESENTATION")

DATA_TCRpep_SPLIT = Processor.DATA_REPRESENTATION(DATA_TRAIN)
DATA_TRAIN_FULL = pd.concat([DATA_TRAIN, DATA_TCRpep_SPLIT], axis=1)
X_TRAIN, y_TRAIN = Processor.fn_downsampling(DATA_TRAIN_FULL)

X_TEST, y_TEST = Processor.DATA_REPRESENTATION(DATA_TEST),  DATA_TEST[["binder"]]

X_TRAIN_cv, y_TRAIN_cv = Processor.cv_data_kd(X_TRAIN), np.squeeze(np.array(y_TRAIN))
X_TEST_cv, y_TEST_cv = Processor.cv_data_kd(X_TEST), np.squeeze(np.array(y_TEST))

###--TRAINING

print("###---TRAINING")

# Create the teacher
teacher = keras.Sequential(
    [
        keras.Input(shape=(17, 4, 1)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),  # Binary classification with sigmoid activation
    ],
    name="teacher",
)

# Create the student
student = keras.Sequential(
    [
        keras.Input(shape=(17, 4, 1)),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),  # Binary classification with sigmoid activation
    ],
    name="student",
)

# Clone student for later comparison
student_scratch = keras.models.clone_model(student)

batch_size = 64

# Train teacher as a binary classifier
teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

# Convert the labels for binary classification
train_labels_binary = y_TRAIN_cv.copy()
test_labels_binary = y_TEST_cv.copy()

# Train and evaluate teacher on data
teacher.fit(X_TRAIN_cv, train_labels_binary, epochs=5)
teacher.evaluate(X_TEST_cv, test_labels_binary)

# Initialize and compile distiller
distiller = KD.Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.BinaryAccuracy()],
    student_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=5,
)

# Convert the labels for binary classification
train_labels_binary = y_TRAIN_cv.copy()
test_labels_binary = y_TEST_cv.copy()

# Distill teacher to student
distiller.fit(X_TRAIN_cv, train_labels_binary, epochs=3)

# Evaluate student on test dataset
distiller.evaluate(X_TEST_cv, test_labels_binary)

# Train student as done usually
student_scratch.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

# Convert the labels for binary classification
train_labels_binary = y_TRAIN_cv.copy()
test_labels_binary = y_TEST_cv.copy()

# Train and evaluate student trained from scratch.
student_scratch.fit(X_TRAIN_cv, train_labels_binary, epochs=3)
student_scratch.evaluate(X_TEST_cv, test_labels_binary)

# student_scratch.save("model.h5")
student_scratch.save(args.savemodel)

###---Evaluation
print("###---EVALUATION")
predicted_probabilities = student_scratch.predict(X_TEST_cv)
predicted_labels = predicted_probabilities.argmax(axis=1)
predicted_labels = (predicted_probabilities >= 0.5).astype(int)

df_label = pd.DataFrame(zip(predicted_probabilities.squeeze(), predicted_labels.squeeze()), columns=['proba_pred', 'binder_pred'])
data_pred = pd.concat([DATA_TEST, df_label], axis=1)

print("###---SAVE DATA")
# data_pred.to_csv(args.outfile, index=False)
data_pred.to_parquet(args.outfile)
