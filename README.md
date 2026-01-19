# Quantum Machine Learning (QML) Classifiers – VQC

Overview

This project demonstrates Quantum Machine Learning (QML) using Variational Quantum Classifiers (VQC) and compares their performance with classical machine learning models.

We use two datasets:
* Credit Card Fraud Detection – 8-feature dataset for financial fraud classification.
* Breast Cancer Wisconsin Dataset (scikit-learn) – standard medical diagnosis dataset with 30 numerical features

The goal is to evaluate the performance of quantum classifiers in terms of accuracy, AUC-ROC, confusion matrix, and classification reports, and benchmark against classical models like Logistic Regression, Random Forest, and SVM.
The objective is not to claim quantum advantage, but to systematically evaluate how VQC models perform under ideal and noisy quantum simulations, and compare them with classical baselines using consistent evaluation metrics.

## Project Motivation

* Classical machine learning models are highly effective, but quantum machine learning models offer the potential to:
* Handle complex, high-dimensional, or non-convex datasets efficiently.
* Provide quantum advantages in pattern recognition for NISQ (Noisy Intermediate-Scale Quantum) devices.
* Explore hybrid quantum-classical architectures combining quantum circuits with classical post-processing.

## Key Steps in the Notebook

### 1. Data Preprocessing

* Load Dataset: Read CSV files using pandas.
* Separate Features and Target: Target column is fraud (or diagnosis for breast cancer).
* Numerical Features: Scaled using StandardScaler.
* Binary/Categorical Features: Reduced using PCA to 1 dimension for quantum compatibility.
* Combine Features: Final X_reduced is quantum-friendly.
* Train-Test Split: Split dataset into training and testing sets (80/20).

### 2. Classical Baseline Models

* Logistic Regression
* Random Forest Classifier
* Support Vector Machine (SVM)

Steps:

* Train each model on X_train.
* Predict probabilities and labels on X_test.
* Compute AUC-ROC, Accuracy, and store probabilities for ROC curve comparisons.

### 3. Variational Quantum Classifier (VQC)

* Feature Map: ZZFeatureMap encodes classical data into quantum states.
* Ansatz: RealAmplitudes – parameterized circuit with rotation and entanglement layers.
* Sampler: StatevectorSampler for ideal simulations.
* Optimizer: COBYLA – gradient-free optimizer.

* Training:
Fit VQC on X_train and y_train.
Predict probabilities and labels on X_test.

### 4. Noisy VQC Simulation

* Use AerSimulator and SamplerV2 to simulate realistic noisy quantum hardware.
* Apply a PassManager to optimize and transpile circuits for the simulator.
* Fit VQC on a smaller subset of data (for speed).
* Evaluate predictions on X_test.

### 5. Evaluation Metrics

For all models:

* Accuracy – Correct predictions ratio.
* AUC-ROC – Area under the ROC curve for probability predictions.
* Confusion Matrix – Visualize true positives/negatives and false positives/negatives.
* Classification Report – Precision, recall, f1-score for each class.
* ROC Curve – Compare all models visually.

### 6. Visualization

* Bar plots for benchmarking metrics across classical and quantum models.
* Confusion matrices for VQC Ideal and VQC Noisy.
* ROC curves for all models to visualize prediction quality.

### 7. Observations:

* Classical models still outperform the small VQC due to dataset size and low qubit count.
* Noisy VQC demonstrates robustness in prediction despite noise.
* Quantum classifiers are trainable and evaluatable on standard datasets using simulators.

## Requirements
```
pip install numpy pandas matplotlib scikit-learn qiskit qiskit-aer qiskit-machine-learning qiskit-ibm-runtime pylatexenc
```
## How to Run

* Clone the repository.
* Open the Jupyter notebook (.ipynb).
* Run each cell sequentially.
* Ensure datasets are in the same folder as the notebook.
* The notebook visualizes quantum circuits for feature map and ansatz using mpl and text.
* Noisy simulations are included for NISQ-style benchmarking.

## References

* Qiskit Machine Learning Documentation
* Schuld, M. et al., Supervised Learning with Quantum Computers, 2019
* scikit-learn Breast Cancer Wisconsin Dataset
* Kaggle Credit Card Fraud Dataset
