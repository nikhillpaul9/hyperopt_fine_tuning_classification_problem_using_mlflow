# Automated Multi-Architecture ML Tuning Pipeline 🚀

An end-to-end, production-ready machine learning pipeline for classification tasks. This project automates data preprocessing, multi-architecture hyperparameter tuning, model ensembling, and experiment tracking using **Hyperopt** and **MLflow**. 

It supports both **local SQLite tracking** and **cloud-based tracking via DagsHub**. While currently configured for the Breast Cancer Wisconsin (Diagnostic) dataset, the modular architecture allows it to be seamlessly adapted to any tabular classification problem.

## ✨ Key Features

* **Multi-Architecture Search Space:** Simultaneously trains and evaluates distinct algorithm families within a single optimization loop:
  * Random Forest
  * XGBoost
  * AdaBoost
  * Artificial Neural Networks (MLP)
* **Meta-Learning / Stacking:** Dynamically constructs a Stacked Ensemble combining the optimized base estimators using a Logistic Regression meta-learner.
* **Bayesian Hyperparameter Tuning:** Leverages Hyperopt's Tree-structured Parzen Estimator (TPE) algorithm to efficiently search the parameter space and maximize classification **Accuracy**.
* **Hybrid Experiment Tracking:** Track your metrics, parameters, and models locally using an MLflow SQLite backend, or push them securely to the cloud using **DagsHub**.
* **Automated Visual Diagnostics:** Automatically generates and logs high-quality matplotlib/seaborn artifacts locally and to MLflow, including:
  * Target Distribution & Feature Correlation Heatmaps (EDA)
  * Trial-specific Confusion Matrices
  * ROC Curves (with AUC scores)
  * Feature Importance Mappings (for tree-based architectures)

## 🛠️ Technology Stack

* **Machine Learning:** `scikit-learn`, `xgboost`
* **Optimization:** `hyperopt`
* **Experiment Tracking & Cloud:** `mlflow`, `dagshub`
* **Data Manipulation & Visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`

## 📂 Project Structure

Upon execution, the pipeline maps visualizations to your local directory and logs data to your chosen backend:

```text
├── mlruns.db                       # MLflow SQLite backend database (Local mode)
├── mlruns/                         # Serialized model binaries and MLflow metadata
├── local_project_graphs/
│   ├── eda/                        # Global dataset EDA plots
│   └── evaluation/                 # Sub-folders for every Hyperopt trial
│       ├── Trial_1_xgboost/
│       │   ├── confusion_matrix.png
│       │   ├── roc_curve.png
│       │   └── feature_importance.png
│       └── Trial_2_stacked_ensemble/ ...
├── main.py                         # Standard pipeline script (Local Tracking)
└── main_dagshub.py                 # Cloud pipeline script (DagsHub Tracking + Headless Matplotlib)
