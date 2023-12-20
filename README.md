# epiTCR-KDA: Knowledge Distillation model on Dihedral Angles for TCR-peptide prediction


This repository contains the code and the data to train [epiTCR-KDA](https://github.com/ddiem-ri-4D/epiTCR-KDA) model.

## Requirements

+ Python >= 3.6.8
+ Keras 2.6.0
+ TensorFlow 2.6.0

## Steps to train a Binding Prediction model for TCR-peptide pairs

### 1. Clone the repository
```bash
git clone https://github.com/ddiem-ri-4D/epiTCR-KDA
cd epiTCR-KDA/
conda create --name kda python=3.6.8
pip3 install pandas==2.0.3 tensorflow==2.13.0 keras==2.13.1 scikit-learn==1.1.2
source activate kda
```

### 2. How to run epiTCR-KDA
- Download training and testing data from [`datasets`](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets) folder.
- Obtain weights for TCR and peptides from [`models`](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/models) folder.

### 3. Check if TCR/pep sequences are present in the PDB folders.
- Prepare a list containing unique TCR/peptides from the data for training/testing.
- Check if the unique TCR/peptides are already present in the PDB folders or not:
+ If they are already complete, proceed to step 4.
+ If not, run the 3D structure using [OmegaFold](https://github.com/HeliXonProtein/OmegaFold) and add the structure to the [PDB folders](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets/3DS_PDBFiles), following the steps below:

#### 3.1 Run OmegaFold
- Prepare a FASTA file containing the TCR/peptide sequences to run OmegaFold, see an example [here](https://github.com/ddiem-ri-4D/epiTCR-KDA/blob/main/datasets/DATA_4RUN/INPUT_FILE.fasta).
- Refer to the OmegaFold running steps [here](https://github.com/HeliXonProtein/OmegaFold), and place the output into the [PDB files](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets/3DS_PDBFiles) directory.
- Double-check for any TCR/peptides that might still lack a structure. If all structures are present, proceed to step 3.2.
- Run OmegaFold by executing the following command:

```bash
cd utils
python3 checkHavePDB.py 
```

#### 3.2 Run Biopython
- After obtaining the 3D structure, run [Biopython](https://biopython.org/docs/dev/api/Bio.PDB.internal_coords.html) to retrieve Dihedral Angles information, resulting in an output *.tsv file, see an example [here](https://github.com/ddiem-ri-4D/epiTCR-KDA/blob/main/datasets/DA_TSVFiles/TPRVTGGGAM.tsv).
- The output *.tsv files containing Dihedral Angles information are placed into the [DA folders](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets/DA_TSVFiles) directory.

```bash
python3 PDB2DA.py
```

### 4. Retraining and predict model
An example for training and testing

```bash
python3 train.py 
        --trainfile train.parquet \
        --testfile test.parquet \
        --savemodel savemodel.h5 \
        --outfile predict.parquet
```

```bash
python3 test.py 
        --testfile test.parquet \
        --savedmodel savedmodel.h5 \
        --outfile predict.parquet
```

### 5. Contact
For more questions or feedback, please simply post an [Issue](https://github.com/ddiem-ri-4D/epiTCR-KDA/issues/new).

### 6. Citation
Please cite this paper if it helps your research:
