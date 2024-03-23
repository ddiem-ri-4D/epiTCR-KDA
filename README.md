# epiTCR-KDA: Knowledge Distillation model on Dihedral Angles for TCR-peptide prediction


This repository contains the code and the data to train [epiTCR-KDA](https://github.com/ddiem-ri-4D/epiTCR-KDA) model.

## Requirements

+ Python >= 3.6.8
+ Keras 2.6.0
+ TensorFlow 2.6.0

## How to run epiTCR-KDA

### 1. Clone the repository
```bash
git clone https://github.com/ddiem-ri-4D/epiTCR-KDA
cd epiTCR-KDA/
conda create --name kda python=3.6.8
pip3 install pandas==2.0.3 tensorflow==2.13.0 keras==2.13.1 scikit-learn==1.1.2
source activate kda
```

### 2. Prepare data
- Download training and testing data from [`datasets`](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets/DATA_4MODEL) folder.
- Download the 3D structure and dihedral angles of TCR and peptide from folders [`3DS_PDBFiles`](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets/3DS_PDBFiles) and [`DA_TSVFiles`](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets/DA_TSVFiles).

### 3. Check if TCR/pep sequences are present in the PDB folders
- Prepare a list containing unique TCR/peptides from the data for training/testing.
- Check if the unique TCR/peptides are already present in the PDB folders or not by executing the following command:

```bash
cd utils
python3 ./utils/checkHavePDB.py 
```

+ If they are already complete, proceed to step 4.
+ If not, run the 3D structure using [OmegaFold](https://github.com/HeliXonProtein/OmegaFold) and add the structure to the [PDB folders](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets/3DS_PDBFiles), following the steps below:

#### 3.1 Run OmegaFold
- Prepare a FASTA file containing the TCR/peptide sequences to run OmegaFold, see an example [here](https://github.com/ddiem-ri-4D/epiTCR-KDA/blob/main/datasets/DATA_4RUN/INPUT_FILE.fasta).
- Refer to the OmegaFold running steps [here](https://github.com/HeliXonProtein/OmegaFold), and place the output into the [PDB files](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets/3DS_PDBFiles) directory.
- Double-check for any TCR/peptides that might still lack a structure. If all structures are present, proceed to step 3.2.

#### 3.2 Run Biopython
- After obtaining the 3D structure, run [Biopython](https://biopython.org/docs/dev/api/Bio.PDB.internal_coords.html) to retrieve Dihedral Angles information, resulting in an output *.tsv file, see an example [here](https://github.com/ddiem-ri-4D/epiTCR-KDA/blob/main/datasets/DA_TSVFiles/AAFKGAQKLV.tsv).
- The output *.tsv files containing Dihedral Angles information are placed into the [DA folders](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets/DA_TSVFiles) directory.

```bash
python3 ./utils/PDB2DA.py
```

### 4. Retraining and predict model
An example for training and testing

```bash
python3 train.py \
        --trainfile ./datasets/DATA_4MODEL/TRAIN-TEST/train.parquet \
        --testfile ./datasets/DATA_4MODEL/TRAIN-TEST/test.parquet \
        --savemodel ./models/KDA_model.h5 \
        --outfile ./datasets/DATA_4PRED/test_prediction.parquet
```

```bash
python3 test.py \
        --testfile ./datasets/DATA_4MODEL/TRAIN-TEST/test.parquet \
        --savedmodel ./models/KDA_model.h5 \
        --outfile ./datasets/DATA_4PRED/test_prediction.parquet
```

### 5. Contact
For more questions or feedback, please simply post an [Issue](https://github.com/ddiem-ri-4D/epiTCR-KDA/issues/new).

### 6. Citation
Please cite this paper if it helps your research:
