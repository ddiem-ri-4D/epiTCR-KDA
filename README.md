# epiTCR-KDA: Knowledge Distillation model on Dihedral Angles for TCR-peptide prediction


## Publication

## Dependencies

+ Python >= 3.6.8
+ Keras 2.6.0
+ TensorFlow 2.6.0

## Steps to train a Binding Prediction model for TCR-peptide pairs

### 1. Clone the repository
```bash
git clone https://github.com/ddiem-ri-4D/epiTCR-KDA
cd epiTCR-KDA/
conda create --name kda python=3.6.13
pip3 install pandas==2.0.3 tensorflow==2.13.0 keras==2.13.1 scikit-learn==1.1.2
source activate kda
```

### 2. Prepare TCR-peptide pairs for training and testing
- Download training and testing data from `datasets` folder.
- Obtain weights for TCR and peptides from `models` folder.

### 3. Check if TCR/pep sequences are present in the PDB folders.
- If the structure already exists, proceed to step 4.
- If not, run the 3D structure using [OmegaFold](https://github.com/HeliXonProtein/OmegaFold) and add the executed structure to the PDB folders, following these steps:

### 3.1 Run OmegaFold

- Prepare a list containing the TCR/peptides for 3D structure modeling.
- Run OmegaFold by executing the following command:

```bash
cd utils
python3 checkHavePDB.py 
```

### 3.2 Run Biopython

- After obtaining the 3D structure, run Biopython to retrieve Dihedral Angles information, resulting in an output *tsv file.

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
