# Automated Multi-Architecture ML Tuning Pipeline 🚀

An end-to-end, production-ready machine learning pipeline for classification tasks. This project automates the data preprocessing, multi-architecture hyperparameter tuning, model ensembling, and experiment tracking using **Hyperopt** and **MLflow**.

While currently configured for the Breast Cancer Wisconsin (Diagnostic) dataset, the modular architecture allows it to be seamlessly adapted to any tabular classification problem.

## ✨ Key Features

* **Multi-Architecture Search Space:** Simultaneously trains and evaluates distinct algorithm families within a single optimization loop:
  * Random Forest
  * XGBoost
  * AdaBoost
  * Artificial Neural Networks (MLP)
* **Meta-Learning / Stacking:** Dynamically constructs a Stacked Ensemble combining the optimized base estimators using a Logistic Regression meta-learner.
* **Bayesian Hyperparameter Tuning:** Leverages Hyperopt's Tree-structured Parzen Estimator (TPE) algorithm to efficiently search the parameter space and maximize classification **Accuracy**.
* **Comprehensive Experiment Tracking:** Integrates an MLflow tracking server (SQLite backend) to log all parameters, metrics, and serialized model signatures for full reproducibility.
* **Automated Visual Diagnostics:** Automatically generates and logs high-quality matplotlib/seaborn artifacts locally and to MLflow, including:
  * Target Distribution & Feature Correlation Heatmaps (EDA)
  * Trial-specific Confusion Matrices
  * ROC Curves (with AUC scores)
  * Feature Importance Mappings (for tree-based architectures)

## 🛠️ Technology Stack

* **Machine Learning:** `scikit-learn`, `xgboost`
* **Optimization:** `hyperopt`
* **Experiment Tracking & Serialization:** `mlflow`
* **Data Manipulation & Visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`

## 📂 Artifact Generation Structure

Upon execution, the pipeline automatically spins up an SQLite database and maps visualizations to your local directory:

```text
├── mlruns.db                       # MLflow SQLite backend database
├── mlruns/                         # Serialized model binaries and MLflow metadata
├── local_project_graphs/
│   ├── eda/                        # Global dataset EDA plots
│   └── evaluation/                 # Sub-folders for every Hyperopt trial
│       ├── Trial_1_xgboost/
│       │   ├── confusion_matrix.png
│       │   ├── roc_curve.png
│       │   └── feature_importance.png
│       └── Trial_2_stacked_ensemble/ ...
└── main.py                         # Core pipeline script
