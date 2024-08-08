import pandas as pd
import numpy as np
import os
from imblearn.under_sampling import RandomUnderSampler

def data_representation(data):
    cdr3_data = extract_protein_dihedral(data.CDR3b.unique(), "./datasets/DA_TSVFiles/")
    epitope_data = extract_protein_dihedral(data.epitope.unique(), "./datasets/DA_TSVFiles/")

    combined_data = create_dataframe(data, cdr3_data, epitope_data)
    combined_data_split = combined_data[[f'T{i}' for i in range(1, 35)] + [f'E{i}' for i in range(1, 19)]]
    
    return combined_data_split

def extract_protein_dihedral(sequence_list, folder_path):
    protein_data = {}
    skipped_sequences = []

    for sequence in sequence_list:
        file_path = os.path.join(folder_path, f"{sequence}.tsv")
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        if os.path.getsize(file_path) == 0:
            print(f"File is empty: {file_path}")
            skipped_sequences.append(sequence)
            continue
        
        try:
            df = pd.read_csv(file_path, delimiter='\t', header=None, names=['residueID', 'X_phi', 'Y_psi', 'label'])
            df = df[['X_phi', 'Y_psi']]
            protein_data[sequence] = df.values.flatten().tolist()
        except pd.errors.EmptyDataError:
            print(f"EmptyDataError: Skipping {file_path}")
            skipped_sequences.append(sequence)

    if skipped_sequences:
        print(f"Skipped sequences: {', '.join(skipped_sequences)}")

    return {k: v for k, v in protein_data.items() if v is not None}

def downsample_data(data):
    features = data[[f'T{i}' for i in range(1, 35)] + [f'E{i}' for i in range(1, 19)]]
    labels = data[["binder"]]

    undersampler = RandomUnderSampler(random_state=42)
    resampled_features, resampled_labels = undersampler.fit_resample(features, labels)
    resampled_features.reset_index(drop=True, inplace=True)
    resampled_labels.reset_index(drop=True, inplace=True)
    
    return resampled_features, resampled_labels

def vector_dict_to_df(vector_dict, prefix, num_columns):
    vector_dict = {key: value for key, value in vector_dict.items() if value is not None}
    
    df = pd.DataFrame.from_dict(vector_dict, orient='index')
    df.columns = [f"{prefix}{i}" for i in range(1, df.shape[1] + 1)]
    
    for i in range(len(df.columns) + 1, num_columns + 1):
        df[f"{prefix}{i}"] = 0
    
    return df

def create_dataframe(sample, cdr3_data, epitope_data):
    cdr3_df = vector_dict_to_df(cdr3_data, "T", 34).reset_index().rename(columns={"index": "CDR3b"})
    epitope_df = vector_dict_to_df(epitope_data, "E", 18).reset_index().rename(columns={"index": "epitope"})

    merged_data = sample.merge(cdr3_df, how='left', on='CDR3b')
    merged_data = merged_data.merge(epitope_df, how='left', on='epitope')
    
    merged_data[[f'T{i}' for i in range(1, 35)]] = merged_data[[f'T{i}' for i in range(1, 35)]].fillna(0)
    merged_data[[f'E{i}' for i in range(1, 19)]] = merged_data[[f'E{i}' for i in range(1, 19)]].fillna(0)
    final_data = merged_data.dropna(subset=['T1', 'E1']).copy()
    
    return final_data

def prepare_cv_data(features):
    cdr3_features = features.iloc[:, :34].values.reshape((len(features), 17, 2))
    epitope_features = features.iloc[:, 34:].values.reshape((len(features), 9, 2))

    combined_features = []
    padding = np.zeros((cdr3_features.shape[1] - epitope_features.shape[1], epitope_features.shape[2]))
    for cdr3, epitope in zip(cdr3_features, epitope_features):
        epitope_padded = np.vstack((epitope, padding))
        combined = np.concatenate((cdr3, epitope_padded), axis=1)
        combined_features.append(combined)

    return np.expand_dims(np.array(combined_features), axis=-1)

def list_unseen_epitopes(train_data, test_data):
    train_epitopes = set(train_data.epitope.unique())
    test_epitopes = set(test_data.epitope.unique())
    
    unseen_epitopes = test_epitopes - train_epitopes
    return list(unseen_epitopes), len(unseen_epitopes)

def validate_epitope_length(df):
    invalid_chars = r"[*_\-O1ylX/ #(?]"
    df = df[~df.Antigen.str.contains(invalid_chars)]
    df["len_epitope"] = df.Antigen.str.len()
    df = df[(df["len_epitope"] <= 11) & (df["len_epitope"] >= 8)]
    df.drop(['len_epitope'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def process_sequence(sequence):
    return sequence[1:-1] if sequence.startswith('C') and sequence.endswith('F') else sequence

def validate_tcr_length(df):
    invalid_chars = r"[*_\-O1ylX/ #(?]"
    df = df[~df["CDR3b"].str.contains(invalid_chars)]
    df["CDR3b"] = df["CDR3b"].apply(process_sequence)
    df["len_cdr3"] = df["CDR3b"].str.len()
    df = df[(df["len_cdr3"] <= 19) & (df["len_cdr3"] >= 8)]
    df.drop(['len_cdr3'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def validate_epitope_length(df):
    invalid_chars = r"[*_\-O1ylX/ #(?]"
    df = df[~df["epitope"].str.contains(invalid_chars, na=False)]
    df["len_epi"] = df["epitope"].str.len()
    df = df[(df["len_epi"] <= 11) & (df["len_epi"] >= 8)]
    df.drop(['len_epi'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
