import numpy as np
import pandas as pd
import warnings
import sys
from keras.models import load_model
from argparse import ArgumentParser
import modules.processor as processor

warnings.filterwarnings('ignore')

def main():
    # Argument parsing
    parser = ArgumentParser(description="Specifying Input Parameters")
    parser.add_argument("-te", "--testfile", help="Specify the full path of the file with TCR sequences")
    parser.add_argument("-sm", "--savedmodel", help="Specify saved model file")
    parser.add_argument("-o", "--outfile", default=sys.stdout, help="Specify output file")

    args = parser.parse_args()

    # Load and process the dataset
    print("###---LOADING DATA")
    test_data = pd.read_parquet(args.testfile)
    test_data = processor.validate_tcr_length(test_data)
    test_data = processor.validate_epitope_length(test_data)
    test_data = test_data.reset_index(drop=True)

    print("###---DATA REPRESENTATION")
    X_test = processor.data_representation(test_data)
    X_test_cv = processor.prepare_cv_data(X_test)

    # Load the pre-trained model
    print("###---LOAD MODEL PRETRAINING")
    loaded_model = load_model(args.savedmodel)

    # Evaluation
    print("###---EVALUATION")
    predicted_probabilities = loaded_model.predict(X_test_cv)
    predicted_labels = (predicted_probabilities >= 0.5).astype(int)

    results_df = pd.DataFrame({
        'proba_pred': predicted_probabilities.squeeze(),
        'binder_pred': predicted_labels.squeeze()
    })
    predicted_data = pd.concat([test_data, results_df], axis=1)

    # Save the predictions
    print("###---SAVE DATA")
    predicted_data.to_parquet(args.outfile)

if __name__ == "__main__":
    main()
