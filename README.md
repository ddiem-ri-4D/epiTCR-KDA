# epiTCR-KDA: Knowledge Distillation model on Dihedral Angles for TCR-peptide prediction


This repository contains code and data to train [epiTCR-KDA](https://academic.oup.com/bioinformaticsadvances/advance-article/doi/10.1093/bioadv/vbae190/7914039) model.

## Requirements

+ Python >= 3.6.8
+ Keras 2.6.0
+ TensorFlow 2.6.0

## How to run epiTCR-KDA
![pipeline](https://github.com/ddiem-ri-4D/epiTCR-KDA/blob/main/assets/Figure1_cut.png)

### 1. Clone repository
```bash
git clone https://github.com/ddiem-ri-4D/epiTCR-KDA
cd epiTCR-KDA/
conda env create -f environment.yml
source activate kda
```

### 2. Prepare data
- Download training and testing data from [`datasets`](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets/DATA_4MODEL) folder.
- Download 3D structure and dihedral angles of TCR and peptide from folders [`3DS_PDBFiles`](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets/3DS_PDBFiles) and [`DA_TSVFiles`](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets/DA_TSVFiles).

### 3. Check if TCR/pep sequences are present in DA folders
- Prepare a list containing unique TCR/peptides from data for training/testing.
- Check if unique TCR/peptides are already present in DA_TSVFiles folders or not by executing the following command:

```bash
cd utils
python3 check3DSDA.py 
```

+ If they are already complete, proceed to step 4.
+ If not, run 3D structure using [OmegaFold](https://github.com/HeliXonProtein/OmegaFold) and add structure to [PDB folders](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets/3DS_PDBFiles), following steps below:

#### 3.1 Run OmegaFold
- Prepare a FASTA file containing TCR/peptide sequences to run OmegaFold, see an example [here](https://github.com/ddiem-ri-4D/epiTCR-KDA/blob/main/datasets/DATA_4RUN/INPUT_FILE.fasta).
- Refer to OmegaFold running steps [here](https://github.com/HeliXonProtein/OmegaFold), and place output into [PDB files](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets/3DS_PDBFiles) directory.
- Double-check for any TCR/peptides that might still lack a structure. If all structures are present, proceed to step 3.2.

#### 3.2 Run Biopython
- After obtaining 3D structure, run [Biopython](https://biopython.org/docs/dev/api/Bio.PDB.internal_coords.html) to retrieve Dihedral Angles information, resulting in an output *.tsv file, see an example [here](https://github.com/ddiem-ri-4D/epiTCR-KDA/blob/main/datasets/DA_TSVFiles/AAFKGAQKLV.tsv).
- output *.tsv files containing Dihedral Angles information are placed into [DA folders](https://github.com/ddiem-ri-4D/epiTCR-KDA/tree/main/datasets/DA_TSVFiles) directory.

```bash
cd utils
python3 PDB2DA.py
```

### 4. Retrain and Predict Model
- `train.parquet`/`test.parquet`: input parquet file with 3 columns named as "CDR3b, epitope, binder (if training)": CDR3 sequence, peptide sequence, and CDR3b and peptide bind together or not.

| CDR3b         | epitope       | binder|
| ------------- |:-------------:| -----:|
| AASSYGQNFV    | QIKVRVDMV     | 1     |
| AIRAGGDEQ     | HSKKKCDEL     | 1     |
| AISETDKLG     | LPPIVAKEI     | 1     |
| SARDRVRTDTQY  | FVSKLYYFE     | 0     |
| SARDRVRTDTQY  | KLSHQPVLL     | 0     |

- An example for training and testing
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
For more questions or feedback, please post an [Issue](https://github.com/ddiem-ri-4D/epiTCR-KDA/issues/new).

### 6. Citation
Please cite this paper if it helps your research:
```bibtex
@article{10.1093/bioadv/vbae190,
    author = {Pham, My-Diem Nguyen and Su, Chinh Tran-To and Nguyen, Thanh-Nhan and Nguyen, Hoai-Nghia and Nguyen, Dinh Duy An and Giang, Hoa and Nguyen, Dinh-Thuc and Phan, Minh-Duy and Nguyen, Vy},
    title = {epiTCR-KDA: Knowledge Distillation model on Dihedral Angles for TCR-peptide prediction},
    journal = {Bioinformatics Advances},
    pages = {vbae190},
    year = {2024},
    month = {11},
    issn = {2635-0041},
    doi = {10.1093/bioadv/vbae190},
    url = {https://doi.org/10.1093/bioadv/vbae190},
    eprint = {https://academic.oup.com/bioinformaticsadvances/advance-article-pdf/doi/10.1093/bioadv/vbae190/60923941/vbae190.pdf},}
```
### 7. Reference

My-Diem Nguyen Pham, Thanh-Nhan Nguyen, Le Son Tran, Que-Tran Bui Nguyen, Thien-Phuc Hoang Nguyen, Thi Mong Quynh Pham, Hoai-Nghia Nguyen, Hoa Giang, Minh-Duy Phan, Vy Nguyen, epiTCR: a highly sensitive predictor for TCR–peptide binding, Bioinformatics, Volume 39, Issue 5, May 2023, btad284, https://doi.org/10.1093/bioinformatics/btad284
