README

Static Anomaly Detection using Machine Learning in 5G-enabled IoT Systems

--------------------------------------------------
Project Overview
--------------------------------------------------

This project implements a static anomaly detection framework using a Multi-Layer Perceptron (MLP) model for intrusion detection in 5G-enabled IoT environments using the 5G-NIDD dataset.

The implementation evaluates anomaly detection at three classification levels:

1. Binary Classification
   (Benign vs Attack)

2. Attack Category Classification
   (DoS, Scan, SlowrateDoS)

3. Attack Type Classification
   (Fine-grained attack identification)

Unlike the adaptive version, this implementation performs model training once and evaluates performance without dynamic updating or adaptation during inference.

--------------------------------------------------
Implemented Components
--------------------------------------------------

Data preprocessing:
- Missing value handling
- One-hot encoding
- Label encoding
- Data normalization

Feature engineering:
- ANOVA F-score feature selection
- Top 10 feature selection

Model:
- Multi-Layer Perceptron (MLP)

Optimization:
- Hyperparameter tuning using GridSearchCV
- K-Fold Cross Validation

Evaluation:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

--------------------------------------------------
Classification Levels
--------------------------------------------------

Level 1 – Binary Classification
Objective:
Detect whether network traffic is benign or malicious.

Level 2 – Attack Category Classification
Objective:
Classify malicious traffic into:

- DoS
- Scan
- SlowrateDoS

Level 3 – Attack Type Classification
Objective:
Identify the exact type of attack.

--------------------------------------------------
Dataset
--------------------------------------------------

Dataset used:
5G-NIDD Dataset

Dataset source:
https://www.kaggle.com/datasets/humera11/5g-nidd-dataset

The dataset is not included due to size limitations.

Download the dataset and place:

Combined.csv

inside Google Drive before executing the notebook.

--------------------------------------------------
Requirements
--------------------------------------------------

Python 3.x

Required packages:

tensorflow
scikit-learn==1.4.2
scikeras
pandas
numpy
matplotlib
seaborn

Install packages:

pip install tensorflow scikit-learn==1.4.2 scikeras pandas numpy matplotlib seaborn

--------------------------------------------------
How to Run
--------------------------------------------------

1. Open the notebook (.ipynb) in Google Colab.

2. Mount Google Drive:

from google.colab import drive
drive.mount('/content/drive')

3. Upload Combined.csv into Google Drive.

4. Update dataset path if required.

5. Run notebook cells sequentially from top to bottom.

--------------------------------------------------
Output
--------------------------------------------------

Execution produces:

- Trained models
- Classification reports
- Confusion matrices
- Performance visualizations

--------------------------------------------------
Notes
--------------------------------------------------

This implementation was developed for academic and research purposes as part of the master's thesis.

Results may vary slightly due to random initialization and execution environment.
