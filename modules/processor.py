import pandas as pd
import numpy as np
import os
from imblearn.under_sampling import RandomUnderSampler

def DATA_REPRESENTATION(data):
    cdr3_data = getProteinByDiheral(data.CDR3b.unique(), "./datasets/DA_TSVFiles/")
    epitope_data = getProteinByDiheral(data.epitope.unique(), "./datasets/DA_TSVFiles/")

    combined_data = DAtoDataFrame(data, cdr3_data, epitope_data)
    combined_data_split = combined_data[[f'T{i}' for i in range(1, 35)] + [f'E{i}' for i in range(1, 19)]]
    
    return combined_data_split

def getProteinByDiheral(sequence_list, folder_path):
    protein_data = dict.fromkeys(sequence_list)
    skipped_sequences = []

    for sequence in sequence_list:
        file_path = os.path.join(folder_path, sequence + ".tsv")
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        if os.path.getsize(file_path) == 0:
            print(f"File is empty: {file_path}")
            skipped_sequences.append(sequence)
            continue
        
        try:
            df = pd.read_csv(file_path, delimiter='\t', header=None)
            df.columns = ['residueID', 'X_phi', 'Y_psi', 'label']
            df = df[['X_phi', 'Y_psi']]
            protein_data[sequence] = df.values.flatten().tolist()
        except pd.errors.EmptyDataError:
            print(f"EmptyDataError: Skipping {file_path}")
            skipped_sequences.append(sequence)

    if skipped_sequences:
        print(f"Skipped sequences: {', '.join(skipped_sequences)}")

    return {k: v for k, v in protein_data.items() if v is not None}

def fn_downsampling(data):
    features = data[[f'T{i}' for i in range(1, 35)] + [f'E{i}' for i in range(1, 19)]]
    labels = data[["binder"]]

    undersampler = RandomUnderSampler(random_state=42)
    resampled_features, resampled_labels = undersampler.fit_resample(features, labels)
    resampled_features, resampled_labels = resampled_features.reset_index(drop=True), resampled_labels.reset_index(drop=True)
    
    return resampled_features, resampled_labels

def cvVectorDict2DF(vector, prefix, num_columns):
    vector = {key: value for key, value in vector.items() if value is not None}
    
    df = pd.DataFrame.from_dict(vector, orient='index')
    df.columns = [f"{prefix}{i}" for i in range(1, df.shape[1] + 1)]
    
    for i in range(len(df.columns) + 1, num_columns + 1):
        df[f"{prefix}{i}"] = 0
    
    return df

def DAtoDataFrame(sample, cdr3_data, epitope_data):
    cdr3_df = cvVectorDict2DF(cdr3_data, "T", 34)
    epitope_df = cvVectorDict2DF(epitope_data, "E", 18)
    
    cdr3_df = cdr3_df.reset_index().rename(columns={"index": "CDR3b"})
    epitope_df = epitope_df.reset_index().rename(columns={"index": "epitope"})

    merged_data = sample.merge(cdr3_df, how='left', on='CDR3b')
    merged_data = merged_data.merge(epitope_df, how='left', on='epitope')
    
    merged_data[[f'T{i}' for i in range(1, 35)]] = merged_data[[f'T{i}' for i in range(1, 35)]].fillna(0)
    merged_data[[f'E{i}' for i in range(1, 19)]] = merged_data[[f'E{i}' for i in range(1, 19)]].fillna(0)
    final_data = merged_data.dropna(subset=['T1', 'E1']).copy()
    
    return final_data

def cv_data_kd(test_features):
    cdr3_features = test_features.iloc[:, :34].values.reshape((len(test_features), 17, 2))
    epitope_features = test_features.iloc[:, 34:].values.reshape((len(test_features), 9, 2))

    combined_features = []
    nan_array = np.full((cdr3_features.shape[1] - epitope_features.shape[1], epitope_features.shape[2]), 0)
    for cdr3, epitope in zip(cdr3_features, epitope_features):
        combined = np.concatenate((cdr3, np.concatenate((epitope, nan_array), axis=0)), axis=1)
        combined_features.append(combined)

    return np.expand_dims(np.array(combined_features), axis=-1)

def fn_lst_unseen(train_data, test_data):
    train_epitopes = train_data.epitope.unique().tolist()
    test_epitopes = test_data.epitope.unique().tolist()
    
    unseen_epitopes = [epitope for epitope in test_epitopes if epitope not in train_epitopes]
    return unseen_epitopes, len(unseen_epitopes)

def check_length_epitope(df):
    invalid_chars = ["\*", '_', '-', 'O', '1', 'y', 'l', 'X', '/', ' ', '#', '\(', '\?']
    df = df[~df.Antigen.str.contains('|'.join(invalid_chars))]
    df["len_epitope"] = df.Antigen.str.len()
    df = df[(df["len_epitope"] <= 11) & (df["len_epitope"] >= 8)]
    df = df.drop(['len_epitope'], axis=1).reset_index(drop=True)
    return df

def process_sequence(sequence):
    return sequence[1:-1] if sequence.startswith('C') and sequence.endswith('F') else sequence

def check_length_tcr(df):
    invalid_chars = ["\*", '_', '-', 'O', '1', 'y', 'l', 'X', '/', ' ', '#', '\(', '\?']
    df = df[~df["CDR3b"].str.contains('|'.join(invalid_chars))]
    df["CDR3b"] = df["CDR3b"].apply(process_sequence)
    df["len_cdr3"] = df["CDR3b"].str.len()
    df = df[(df["len_cdr3"] <= 19) & (df["len_cdr3"] >= 8)].drop(['len_cdr3'], axis=1).reset_index(drop=True)
    return df

def check_length_epi(df):
    invalid_chars = ["\*", '_', '-', 'O', '1', 'y', 'l', 'X', '/', ' ', '#', '\(', '\?']
    df = df[~df["epitope"].str.contains('|'.join(invalid_chars), na=False)]
    df["len_epi"] = df["epitope"].str.len()
    df = df[(df["len_epi"] <= 11) & (df["len_epi"] >= 8)].drop(['len_epi'], axis=1).reset_index(drop=True)
    return df
