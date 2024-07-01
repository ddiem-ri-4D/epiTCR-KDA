import pandas as pd
import numpy as np
import os
from imblearn.under_sampling import RandomUnderSampler

def DATA_REPRESENTATION(DATA):
    DATA_cdr3 = getProteinByDiheral(DATA.CDR3b.unique(), "./datasets/DA_TSVFiles/")
    DATA_pep = getProteinByDiheral(DATA.epitope.unique(), "./datasets/DA_TSVFiles/")

    DATA_TCRpep = DAtoDataFrame(DATA, DATA_cdr3, DATA_pep)
    DATA_TCRpep_SPLIT = DATA_TCRpep[[f'T{i}' for i in range(1, 35)] + [f'E{i}' for i in range(1, 19)]]
    
    return DATA_TCRpep_SPLIT

def getProteinByDiheral(list_seq, link):
    folder_path = link
    dict_lst = dict.fromkeys(list_seq)
    skipped_sequences = []

    for key, _ in dict_lst.items():
        csv_file = os.path.join(folder_path, key + ".tsv")
        
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f"File does not exist: {csv_file}")
        
        if os.path.getsize(csv_file) == 0:
            print(f"File is empty: {csv_file}")
            skipped_sequences.append(key)
            continue
        
        # Process the file
        try:
            df = pd.read_csv(csv_file, delimiter='\t', header=None)
            df.columns = ['residueID', 'X_phi', 'Y_psi', 'label']
            df = df[['X_phi', 'Y_psi']]
            values = df.values.flatten().tolist()
            dict_lst[key] = values
        except pd.errors.EmptyDataError:
            print(f"EmptyDataError: Skipping {csv_file}")
            skipped_sequences.append(key)

    if skipped_sequences:
        print(f"Skipped sequences: {', '.join(skipped_sequences)}")

    return {k: v for k, v in dict_lst.items() if v is not None}

def fn_downsampling(data):
    X_train = data[[f'T{i}' for i in range(1, 35)] + [f'E{i}' for i in range(1, 19)]]
    y_train = data[["binder"]]

    nm = RandomUnderSampler(random_state=42)
    X_res, y_res = nm.fit_resample(X_train, y_train)
    X_res, y_res = X_res.reset_index(drop=True), y_res.reset_index(drop=True)
    
    return X_res, y_res

def cvVectorDict2DF(vector, prefix, num_columns):
    vector = {key: value for key, value in vector.items() if value is not None}
    
    temp = pd.DataFrame.from_dict(vector, orient='index')
    temp.columns = [f"{prefix}{i}" for i in range(1, temp.shape[1] + 1)]
    
    for i in range(len(temp.columns) + 1, num_columns + 1):
        temp[f"{prefix}{i}"] = 0
    
    return temp

def DAtoDataFrame(SAMPLE, SAMPLE_CDR3, SAMPLE_PEP):
    SAMPLE_CDR3 = cvVectorDict2DF(SAMPLE_CDR3, "T", 34)
    SAMPLE_PEP = cvVectorDict2DF(SAMPLE_PEP, "E", 18)
    
    SAMPLE_CDR3_df = SAMPLE_CDR3.reset_index().rename(columns={"index": "CDR3b"})
    SAMPLE_PEP_df = SAMPLE_PEP.reset_index().rename(columns={"index": "epitope"})

    SAMPLE_SPLIT_cdr3_merge = SAMPLE.merge(SAMPLE_CDR3_df, how='left', on='CDR3b')
    SAMPLE_SPLIT_pep_merge = SAMPLE_SPLIT_cdr3_merge.merge(SAMPLE_PEP_df, how='left', on='epitope')
    
    columns_to_fill_T = [f'T{i}' for i in range(1, 35)]
    columns_to_fill_E = [f'E{i}' for i in range(1, 19)]
    
    SAMPLE_SPLIT_pep_merge[columns_to_fill_T] = SAMPLE_SPLIT_pep_merge[columns_to_fill_T].fillna(0)
    SAMPLE_SPLIT_pep_merge[columns_to_fill_E] = SAMPLE_SPLIT_pep_merge[columns_to_fill_E].fillna(0)
    SAMPLE_SPLIT_TCRpep = SAMPLE_SPLIT_pep_merge.dropna(subset=['T1', 'E1']).copy()
    
    return SAMPLE_SPLIT_TCRpep

def cv_data_kd(PTEST_X):
    PTEST_X_CDR3 = PTEST_X.iloc[:, :34].values.reshape((len(PTEST_X), 17, 2))
    PTEST_X_epitope = PTEST_X.iloc[:, 34:].values.reshape((len(PTEST_X), 9, 2))

    PTEST_X_cv = []
    nan_array = np.full((PTEST_X_CDR3.shape[1] - PTEST_X_epitope.shape[1], PTEST_X_epitope.shape[2]), 0)
    for cdr3, epitope in zip(PTEST_X_CDR3, PTEST_X_epitope):
        tmp = np.concatenate((cdr3, np.concatenate((epitope, nan_array), axis=0)), axis=1)
        PTEST_X_cv.append(tmp)

    return np.expand_dims(np.array(PTEST_X_cv), axis=-1)

def fn_lst_unseen(data_train, data_test):
    lst_pep_train = data_train.epitope.unique().tolist()
    lst_pep_test = data_test.epitope.unique().tolist()
    
    unseen_epitopes = [item for item in lst_pep_test if item not in lst_pep_train]
    return unseen_epitopes, len(unseen_epitopes)

def check_length_epitope(df):
    discard = ["\*", '_', '-', 'O', '1', 'y', 'l', 'X', '/', ' ', '#', '\(', '\?']
    df = df[~df.Antigen.str.contains('|'.join(discard))]
    df["len_epitope"] = df.Antigen.str.len()
    df = df[(df["len_epitope"] <= 11) & (df["len_epitope"] >= 8)]
    df = df.drop(['len_epitope'], axis=1).reset_index(drop=True)
    return df

def process_sequence(sequence):
    return sequence[1:-1] if sequence.startswith('C') and sequence.endswith('F') else sequence

def check_length_tcr(df):
    discard = ["\*", '_', '-', 'O', '1', 'y', 'l', 'X', '/', ' ', '#', '\(', '\?']
    df = df[~df["CDR3b"].str.contains('|'.join(discard))]
    df["CDR3b"] = df["CDR3b"].apply(process_sequence)
    df["len_cdr3"] = df["CDR3b"].str.len()
    df = df[(df["len_cdr3"] <= 19) & (df["len_cdr3"] >= 8)].drop(['len_cdr3'], axis=1).reset_index(drop=True)
    return df

def check_length_epi(df):
    discard = ["\*", '_', '-', 'O', '1', 'y', 'l', 'X', '/', ' ', '#', '\(', '\?']
    df = df[~df["epitope"].str.contains('|'.join(discard), na=False)]
    df["len_epi"] = df["epitope"].str.len()
    df = df[(df["len_epi"] <= 11) & (df["len_epi"] >= 8)].drop(['len_epi'], axis=1).reset_index(drop=True)
    return df
