import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
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

    dtrain = xgb.DMatrix(X_train_cv.reshape(len(X_train_cv), -1), label=y_train_cv)
    dtest = xgb.DMatrix(X_test_cv.reshape(len(X_test_cv), -1), label=y_test_cv)

    params = {
        'objective': 'binary:logistic',
        'max_depth': 5,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'auc'
    }

    bst = xgb.train(params, dtrain, num_boost_round=100)

    # Save the model
    bst.save_model(args.savemodel)

    preds = bst.predict(dtest)
    predicted_labels = (preds >= 0.5).astype(int)

    xgb_accuracy = accuracy_score(y_test_cv, predicted_labels)
    xgb_auc = roc_auc_score(y_test_cv, preds)

    print(f"XGBoost Accuracy: {xgb_accuracy}")
    print(f"XGBoost AUC: {xgb_auc}")

    print("###---EVALUATION-TEST")

    df_label = pd.DataFrame({
        'proba_pred': preds,
        'binder_pred': predicted_labels
    })
    data_pred = pd.concat([test_data, df_label], axis=1)

    print("###---SAVE DATA TEST")
    data_pred.to_parquet(args.outfiletest)

if __name__ == "__main__":
    main()
