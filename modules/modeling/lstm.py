import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score
from argparse import ArgumentParser
import modules.processor as processor
import warnings
import sys

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

    lstm_model = keras.Sequential([
        keras.Input(shape=(17, 4)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    lstm_model.fit(X_train_cv, y_train_cv, epochs=5, batch_size=64, validation_split=0.2)

    lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test_cv, y_test_cv)
    lstm_auc = roc_auc_score(y_test_cv, lstm_model.predict(X_test_cv))

    print(f"LSTM Accuracy: {lstm_accuracy}")
    print(f"LSTM AUC: {lstm_auc}")

    lstm_model.save(args.savemodel)

    print("###---EVALUATION-TEST")

    predicted_probabilities = lstm_model.predict(X_test_cv)
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
