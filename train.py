import numpy as np
import pandas as pd
import warnings
import os
import sys
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from argparse import ArgumentParser
import modules.architectures as KD
import modules.processor as processor

warnings.filterwarnings('ignore')

def main():
    # Argument parsing
    parser = ArgumentParser(description="Specifying Input Parameters")
    parser.add_argument("-tr", "--trainfile", help="Specify the full path of the training file with TCR sequences")
    parser.add_argument("-te", "--testfile", help="Specify the full path of the file with TCR sequences")
    parser.add_argument("-sm", "--savemodel", help="Specify save model file")
    parser.add_argument("-otest", "--outfiletest", default=sys.stdout, help="Specify output file")

    args = parser.parse_args()

    print("###---LOADING DATA")
    train_data = pd.read_parquet(args.trainfile)
    test_data = pd.read_parquet(args.testfile)
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    train_data = train_data[["CDR3b", "epitope", "binder"]]
    test_data = test_data[["CDR3b", "epitope", "binder"]]

    print("###---DATA REPRESENTATION")
    tcr_epitope_split = processor.data_representation(train_data)
    full_train_data = pd.concat([train_data, tcr_epitope_split], axis=1)
    X_train, y_train = processor.downsample_data(full_train_data)

    X_test, y_test = processor.data_representation(test_data), test_data[["binder"]]

    X_train_cv, y_train_cv = processor.prepare_cv_data(X_train), np.squeeze(np.array(y_train))
    X_test_cv, y_test_cv = processor.prepare_cv_data(X_test), np.squeeze(np.array(y_test))

    print("###---TRAINING")

    # Define teacher model
    teacher_model = keras.Sequential([
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
    ], name="teacher_model")

    # Define student model
    student_model = keras.Sequential([
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
    ], name="student_model")

    # Clone student model for baseline comparison
    student_scratch_model = keras.models.clone_model(student_model)

    batch_size = 64

    # Compile teacher model
    teacher_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    # Stratified cross-validation
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_scores = []

    for train_idx, val_idx in stratified_kfold.split(X_train_cv, y_train_cv):
        X_train_fold, X_val_fold = X_train_cv[train_idx], X_train_cv[val_idx]
        y_train_fold, y_val_fold = y_train_cv[train_idx], y_train_cv[val_idx]

        teacher_model.fit(X_train_fold, y_train_fold, epochs=5, batch_size=batch_size, verbose=1)
        scores = teacher_model.evaluate(X_val_fold, y_val_fold, verbose=1)
        cross_val_scores.append(scores[1])

    # Train teacher model on full data
    teacher_model.fit(X_train_cv, y_train_cv, epochs=5)
    teacher_model.evaluate(X_test_cv, y_test_cv)

    # Initialize and compile distiller
    distiller = KD.Distiller(student=student_model, teacher=teacher_model)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.BinaryAccuracy()],
        student_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=5,
    )

    # Distillation process
    distiller.fit(X_train_cv, y_train_cv, epochs=3)
    distiller.evaluate(X_test_cv, y_test_cv)

    # Train student model from scratch
    student_scratch_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    student_scratch_model.fit(X_train_cv, y_train_cv, epochs=3)
    student_scratch_model.evaluate(X_test_cv, y_test_cv)

    # Save the trained student model
    student_scratch_model.save(args.savemodel)

    print("###---EVALUATION-TEST")

    predicted_probabilities = student_scratch_model.predict(X_test_cv)
    predicted_labels = (predicted_probabilities >= 0.5).astype(int)

    df_label = pd.DataFrame({
        'proba_pred': predicted_probabilities.squeeze(),
        'binder_pred': predicted_labels.squeeze()
    })
    data_pred = pd.concat([test_data, df_label], axis=1)

    print("###---SAVE DATA TEST")
    data_pred.to_parquet(args.outfiletest)

if __name__ == "__main__":
    main()
