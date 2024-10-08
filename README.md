# StackGlyEmbed

### Selected Feature Group
![DeepCPBSite_2 (1)-1](https://github.com/user-attachments/assets/9b84d4a2-f0d3-44c7-ad7c-8fbb337a2279)


### Training framework
![1](https://github.com/nafiislam/StackGlyEmbed/assets/65466905/954c0ee4-c4d2-4a75-91c9-691b5f5de28f)


### Prediction framework
<img src="https://github.com/nafiislam/StackGlyEmbed/assets/65466905/a3f263c7-da44-4c5b-91d6-eda911fd8346" width=50% height=100%>

### Data availability
All training and independent datasets are given in [Dataset](Dataset) folder

### Environments
OS: Ubuntu 22.04.4 LTS

Python version: Python 3.9.19


Used libraries: 
```
numpy==1.26.4
pandas==2.2.1
pytorch==2.2.2
xgboost==2.0.3
pickle5==0.0.11
scikit-learn==1.2.2
matplotlib==3.8.2
PyQt5==5.15.10
imblearn==0.0
skops==0.9.0
shap==0.45.1
IPython==8.18.1
```

### Reproduce results
1. Firstly, download all features. Read the readme.txt of  [all_features](all_features) folder

2. In [N-GlycositeAtlas](N-GlycositeAtlas) and [N-GlyDE](N-GlyDE), all the reproducable codes are given. Also training scripts are also provided. Follow the readme.txt instructions if given in the corresponding folder

3. Lastly, run the main.py file to reproduce results for each tables. Before running, update the path variables inside the file. One exception is [table_7_generation](N-GlycositeAtlas/table_7_generation). You need to first run [save_probabilities.py](N-GlycositeAtlas/table_7_generation/save_probabilities.py) and then run [load_probabilities.py](N-GlycositeAtlas/table_7_generation/load_probabilities.py).

### Prediction
#### Prerequisites
1. You need to have ProteinBert. Follow the following:
```
pip3 install tensorflow tensorflow_addons numpy pandas h5py lxml pyfaidx
git clone https://github.com/nadavbra/protein_bert.git
cd protein_bert
git submodule init
git submodule update
python setup.py install
```
2. transformers, Pytorch and tensorflow are needed for extracting the embeddings.

3. For more query, you can visit the following githubs:

    [ProtT5-XL-U50](https://github.com/agemagician/ProtTrans)

    [ProteinBert](https://github.com/nadavbra/protein_bert)

    [ESM2](https://github.com/facebookresearch/esm)

#### Steps
1. Firsly, you need to fillup [dataset.txt](prediction/dataset.txt). Follow the pattern shown below:

```
Protein_id, site_position_1,site_position_2,...,site_position_n
Fasta
```

2. For predicting N-linked glycosylation sites from a protein sequence, you need to run the [extractFeatures.py](prediction/extractFeatures.py) to generate features and then run [predict.py](prediction/predict.py) for prediction.

### Reproduce previous paper metrics
In [Previous Paper codes](<Previous Paper codes>), scripts are provided for reproducing the results of previous papers.
