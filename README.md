Adaptive 5G Network Intrusion Detection System (5G-NIDD)

Project Overview:
This project implements an adaptive anomaly detection system using a Multi-Layer Perceptron (MLP) neural network on the 5G-NIDD dataset.




--------level 1 Binary Classification--------

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



--------level 2 Multi-class Attack category Classification--------

The system performs:
1)Attack traffic filtering (9 differrent attack types are categorized into 3 different category)
2)Feature selection using ANOVA
3)Class balancing using oversampling and downsampling
4)Multi-class classification using MLP
5)Hyperparameter tuning using GridSearch + K-Fold cross-validation
6)Model evaluation using classification metrics and confusion matrices
The goal is to classify malicious network traffic into major attack categories, providing a middle layer between binary detection and detailed attack-type classification.

Workflow Pipeline
1) Data Preprocessing: Missing values are handled using median filling for numerical features and mode filling for categorical features.
Categorical attributes such as protocol or connection state are converted using one-hot encoding. Binary labels are created to separate benign and attack traffic, and only attack samples are kept for this stage of the pipeline.

2) Attack Type Encoding: The Attack Type column is converted into numerical labels using LabelEncoder. Individual attack types are grouped into broader categories: DoS(ICMPFlood, UDPFlood, SYNFlood, HTTPFlood), Scan(SYNScan, TCPConnectScan, UDPScan) and SlowrateDoS.

3) Feature Selection: Feature importance is calculated using the ANOVA F-test. Top 10 most important features are selected. This reduces dimensionality and improves model efficiency.

4) Train / Validation / Test Split: The dataset is divided into 80% Training, 10% Validation and 10% Test. Stratified sampling is applied to preserve the distribution of attack types across all datasets.

5) Class Balancing: Attack categories may appear with uneven frequencies in the dataset. To prevent model bias minority classes are oversampled and each class is balanced to the same number of sample.

6) Data Normalization: Feature values are standardized using StandardScaler.

7) MLP Model Architecture
The classification model is implemented using a Multi-Layer Perceptron (MLP) neural network.
Architecture:
Input Layer – Selected feature vector
Dense Layer – 32 neurons with ReLU activation
Dropout Layer – Prevents overfitting
Dense Layer – 16 neurons with ReLU activation
Output Layer – Softmax activation for multi-class prediction
The model uses Sparse Categorical Cross-Entropy as the loss function.

8) Hyperparameter Tuning: To optimize model performance, GridSearchCV with 5-fold Stratified Cross-Validation is used.

9) Model Evaluation:The trained model is evaluated on the unseen test dataset.
Performance metrics include: Accuracy,Precision, Recall, F1-score and two confusion matrices are generated.

Objective: The purpose of this stage is to categorize malicious network traffic into broader attack groups, enabling more structured analysis of cyber threats. By grouping attacks into categories such as DoS, Scan, and SlowrateDoS, the system can better understand the general behavior of malicious traffic before performing detailed attack-type classification.






--------level 3 Multi-class Attack type Classification--------

The system performs:
1)Attack traffic filtering ((9 differrent attack types)
2)Feature selection using ANOVA
3)Class balancing using oversampling and downsampling
4)Multi-class classification using MLP
5)Hyperparameter tuning using GridSearch + K-Fold cross-validation
6)Model evaluation using classification metrics and confusion matrices
The goal is to identify the exact type of network attack after traffic has already been detected as malicious.

Workflow Pipeline
1) Data Preprocessing: Missing values are handled using median filling for numerical features and mode filling for categorical features.
Categorical attributes such as protocol or connection state are converted using one-hot encoding. Binary labels are created to separate benign and attack traffic, and only attack samples are kept for this stage of the pipeline.

2) Attack Type Encoding: The Attack Type column is converted into numerical labels using LabelEncoder. Each unique attack type receives an integer label, enabling the neural network to perform multi-class classification.

3) Feature Selection: Feature importance is calculated using the ANOVA F-test. Top 10 most important features are selected. This reduces dimensionality and improves model efficiency.

4) Train / Validation / Test Split: The dataset is divided into 80% Training, 10% Validation and 10% Test. Stratified sampling is applied to preserve the distribution of attack types across all datasets.

5) Class Balancing: Larger classes are downsampled and smaller classes are oversampled. Each attack class is adjusted to a target size of 30,000 samples.

6) Data Normalization: Feature values are standardized using StandardScaler.

7) MLP Model Architecture
The classification model is implemented using a Multi-Layer Perceptron (MLP) neural network.
Architecture:
Input Layer – Selected feature vector
Dense Layer – 32 neurons with ReLU activation
Dropout Layer – Prevents overfitting
Dense Layer – 16 neurons with ReLU activation
Output Layer – Softmax activation for multi-class prediction
The model uses Sparse Categorical Cross-Entropy as the loss function.

8) Hyperparameter Tuning: To optimize model performance, GridSearchCV with 5-fold Stratified Cross-Validation is used.

9) Model Evaluation:The trained model is evaluated on the unseen test dataset.
Performance metrics include: Accuracy,Precision, Recall, F1-score and two confusion matrices are generated.

Objective: The purpose of this stage is to provide fine-grained classification of network attacks, enabling deeper analysis of malicious activity and improving the effectiveness of anomaly detection systems.

------------Requirements------------- 
   (Python 3.x,
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


