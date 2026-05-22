
# Adaptive Hybrid Anomaly Detection System (VAE + XGBoost) for 5G-NIDD

## Project Overview

This project implements an adaptive anomaly detection framework using a Hybrid Variational Autoencoder (VAE) and Extreme Gradient Boosting (XGBoost) model on the 5G-NIDD dataset.

The framework combines deep feature extraction using VAE with efficient classification using XGBoost. In addition, adaptive learning mechanisms are integrated to allow the model to respond to changing network traffic behavior over time.

---

## -------- Level 1 Binary Classification --------

### The system performs:

1. Binary classification (Benign vs Attack)
2. Feature extraction using VAE
3. Classification using XGBoost
4. Hyperparameter tuning
5. Adaptive confidence thresholding
6. Concept drift detection using ADWIN
7. Feedback-based retraining

The goal is to build a hybrid adaptive model capable of detecting malicious traffic while maintaining performance under evolving network conditions.

---

## Workflow Pipeline

### 1) Data Preprocessing

Missing values are handled using median/mode filling.

Categorical features are converted using one-hot encoding.

Binary labels are created:

* 0 = Benign
* 1 = Attack

---

### 2) Feature Engineering and Representation Learning

Input features are normalized using StandardScaler.

A Variational Autoencoder (VAE) is trained to learn compressed latent representations of network traffic.

The encoder output is used as reduced-dimensional feature vectors.

---

### 3) Train / Validation / Test Split

Dataset split:

* 80% Training
* 10% Validation
* 10% Testing

Stratified sampling preserves class distribution.

---

### 4) Class Balancing

Training data is balanced using oversampling to reduce bias toward majority classes.

---

### 5) Hybrid Model Architecture

#### Variational Autoencoder (VAE)

The VAE contains:

* Input Layer
* Encoder Dense Layers
* Latent Space Representation
* Decoder Dense Layers
* Reconstruction Output

The encoder generates compressed representations of network traffic.

#### XGBoost Classifier

The extracted latent features are passed to XGBoost for classification.

XGBoost performs:

* Gradient boosting
* Decision-tree-based classification
* Final attack prediction

---

### 6) Hyperparameter Tuning

Model parameters are optimized using:

* GridSearch
* Cross-validation

Parameters tuned include:

* Latent dimension
* Learning rate
* Tree depth
* Number of estimators
* Regularization parameters

---

### 7) Adaptive System Components

After initial training, the model operates in simulated streaming mode.

Adaptive behavior includes:

* Dynamic confidence threshold adjustment based on recent prediction probabilities
* Sliding window monitoring
* Concept drift detection using ADWIN

When drift is detected:

* Recent labeled samples are collected
* VAE representations are regenerated
* XGBoost is incrementally retrained

This allows adaptation to evolving traffic patterns.

---

### 8) Evaluation

Performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix (raw and normalized)

---

## -------- Level 2 Multi-class Attack Category Classification --------

### The system performs:

1. Attack traffic filtering
2. Feature extraction using VAE
3. Attack category classification using XGBoost
4. Hyperparameter tuning
5. Adaptive learning
6. Model evaluation

Attack categories:

* DoS
* Scan
* SlowrateDoS

Objective:
Classify malicious traffic into broader attack categories.

---

## -------- Level 3 Multi-class Attack Type Classification --------

### The system performs:

1. Attack traffic filtering
2. Feature extraction using VAE
3. Attack type classification using XGBoost
4. Hyperparameter tuning
5. Adaptive learning
6. Model evaluation

Objective:
Identify exact attack types for fine-grained anomaly analysis.

---

## Requirements

Python 3.x

Required libraries:

* TensorFlow
* XGBoost
* Scikit-learn (1.4.2)
* River (ADWIN)
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciKeras

---

## Dataset

Dataset is not included due to size limitations.

Download:
5G-NIDD Dataset

Place:
Combined.csv

inside Google Drive before execution.

---

## How to Run

1. Open notebook in Google Colab

2. Upload dataset to Google Drive

3. Mount Google Drive

4. Run notebook cells sequentially

5. Outputs:

   * Trained models
   * Evaluation metrics
   * Confusion matrices
   * Adaptive learning results
