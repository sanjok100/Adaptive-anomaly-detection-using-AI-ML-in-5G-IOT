Adaptive 5G Network Intrusion Detection System (5G-NIDD)

Project Overview:
This project implements an adaptive anomaly detection system using a Multi-Layer Perceptron (MLP) neural network on the 5G-NIDD dataset.

The system performs:
1) Binary classification (Benign vs Attack)
2) Feature selection using ANOVA
3) Model training with MLP
4) Hyperparameter tuning using GridSearch + K-Fold
5) Adaptive confidence thresholding
6) Concept drift detection using ADWIN
7) Feedback-based retraining
The goal is to build a model that can adapt to changing network traffic patterns over time.

 Workflow Pipeline
1) Data Preprocessing: Missing values are handled (median/mode filling). Categorical features are one-hot encoded. Binary labels are created (0 = Benign, 1 = Attack).

2) Feature Selection: ANOVA F-test is used. Top 10 most important features are selected. This reduces dimensionality and improves model efficiency.

3) Train / Validation / Test Split: 80% training/10% validation/10% test. Stratified splitting is used to preserve class distribution

4) Class Balancing: The training set is balanced using oversampling to prevent bias toward the majority class.

5) Data Normalization: StandardScaler is applied to normalize feature values.

6) MLP Model Architecture: The neural network contains: Input layer, Dense layer (32 neurons, ReLU), Dropout layer, Dense layer (16 neurons, ReLU), Output layer (1 neuron, Sigmoid), Binary cross-entropy loss is used.

7) Hyperparameter Tuning: GridSearchCV with 5-fold cross-validation is used to find the best: Optimizer (Adam / RMSprop), Learning rate, Dropout rate, Batch size

8) Adaptive System Components: After the initial training, the system runs in a simulated streaming mode where predictions are processed one by one. Instead of always using a fixed decision threshold of 0.5, the model adjusts the threshold dynamically based on the recent prediction probabilities stored in a sliding window. If enough recent predictions are available, the threshold is calculated using the mean and standard deviation of those probabilities. That makes the decision boundary adapt to changes in confidence levels. At the same time, the system continuously monitors prediction errors using the ADWIN drift detection algorithm. For each sample, it checks whether the prediction was correct or incorrect and feeds this information to ADWIN. If ADWIN detects a significant change in the error pattern (indicating possible concept drift), the model is fine-tuned using the most recent 500 labeled samples. This allows the system to adjust to evolving network traffic patterns and maintain performance over time.

10) Evaluation: The system reports Accuracy, Classification report (Precision, Recall, F1-score) and Confusion matrix (normalized and raw counts)


Requirements: (Python 3.x,
   TensorFlow,
   Scikit-learn (1.4.2),
   SciKeras,
   River (for ADWIN),
   Matplotlib,
   Seaborn,
   Pandas,
   NumPy)



How to Run:

  Upload dataset (Combined.csv) to Google Drive: The dataset is not included in the repository due to size limitations. Please download the 5G-NIDD dataset and place Combined.csv inside your Google Drive before     running the notebook. 'https://www.kaggle.com/datasets/humera11/5g-nidd-dataset'
  
  Open notebook in Google Colab.
  
  Mount Drive.
  
  Run all cells sequentially.


