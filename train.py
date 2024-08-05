import numpy as np
import pandas as pd
import warnings
import sys
from argparse import ArgumentParser
from keras.models import load_model
from imblearn.under_sampling import RandomUnderSampler
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras import layers
import modules.architectures as KD
import modules.processor as Processor

warnings.filterwarnings('ignore')

# Args parse
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-tr", "--trainfile", help="Specify the full path of the training file with TCR sequences")
parser.add_argument("-te", "--testfile", help="Specify the full path of the file with TCR sequences")
parser.add_argument("-sm", "--savemodel", help="Specify save model file")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="Specify output file")

args = parser.parse_args()

# Load and encode the dataset
print("###---LOADING DATA")
DATA_TRAIN = pd.read_parquet(args.trainfile)
DATA_TEST = pd.read_parquet(args.testfile)

DATA_TRAIN, DATA_TEST = Processor.check_length_tcr(DATA_TRAIN), Processor.check_length_tcr(DATA_TEST)
DATA_TRAIN, DATA_TEST = Processor.check_length_epi(DATA_TRAIN), Processor.check_length_epi(DATA_TEST)
DATA_TRAIN, DATA_TEST = DATA_TRAIN.reset_index(drop=True), DATA_TEST.reset_index(drop=True)

# Data representation
print("###---DATA REPRESENTATION")
DATA_TCRpep_SPLIT = Processor.DATA_REPRESENTATION(DATA_TRAIN)
DATA_TRAIN_FULL = pd.concat([DATA_TRAIN, DATA_TCRpep_SPLIT], axis=1)

X_TRAIN, y_TRAIN = Processor.fn_downsampling(DATA_TRAIN_FULL)
X_TEST, y_TEST = Processor.DATA_REPRESENTATION(DATA_TEST), DATA_TEST[["binder"]]

X_TRAIN_cv, y_TRAIN_cv = Processor.cv_data_kd(X_TRAIN), np.squeeze(np.array(y_TRAIN))
X_TEST_cv, y_TEST_cv = Processor.cv_data_kd(X_TEST), np.squeeze(np.array(y_TEST))

# Training
print("###---TRAINING")
# Create the teacher model
teacher = keras.Sequential(
    [
        keras.Input(shape=(17, 4, 1)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="teacher",
)

# Create the student model with specified layers
student = keras.Sequential(
    [
        keras.Input(shape=(17, 4, 1)),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="student",
)

student_scratch = keras.models.clone_model(student)
batch_size = 64
epochs = 50

# Compile the teacher model
teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

# Compile the student model from scratch
student_scratch.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

# Initialize the distiller
distiller = KD.Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.BinaryAccuracy()],
    student_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=5,
)

# Stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, val_index in skf.split(X_TRAIN_cv, y_TRAIN_cv):
    X_train_fold, X_val_fold = X_TRAIN_cv[train_index], X_TRAIN_cv[val_index]
    y_train_fold, y_val_fold = y_TRAIN_cv[train_index], y_TRAIN_cv[val_index]

    # Train the teacher model
    teacher.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=epochs)
    teacher.evaluate(X_TEST_cv, y_TEST_cv)

    # Distill teacher to student
    distiller.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=epochs)
    distiller.evaluate(X_TEST_cv, y_TEST_cv)

    # Train the student model from scratch
    student_scratch.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=epochs)
    student_scratch.evaluate(X_TEST_cv, y_TEST_cv)

# Save the trained student model
student_scratch.save(args.savemodel)

# Evaluation
print("###---EVALUATION")
predicted_probabilities = student_scratch.predict(X_TEST_cv)
predicted_labels = (predicted_probabilities >= 0.5).astype(int)

df_label = pd.DataFrame({'proba_pred': predicted_probabilities.squeeze(), 'binder_pred': predicted_labels.squeeze()})
data_pred = pd.concat([DATA_TEST, df_label], axis=1)

print("###---SAVE DATA")
data_pred.to_parquet(args.outfile)
