# epiTCR-KDA: Knowledge Distillation model on Dihedral Angles for TCR-peptide prediction


## Publication

## Dependencies

+ Python 3.6.13
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

### 3. Kiểm tra TCR/pep sequences có trong PDB folders hay chưa?
- Nếu có thì tiếp bước 4.
- Nếu chưa thì chạy cấu trúc 3D từ OmegaFold và thêm cấu trúc đã chạy vào PDB folders, thực hiện theo các bước sau:

```bash
cd utils
python3 checkHavePDB.py 
python3 PDB2DA.py
```

### 4. Train and test models
An example for training & testing

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