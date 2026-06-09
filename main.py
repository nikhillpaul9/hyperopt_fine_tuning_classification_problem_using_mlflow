import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc

# Model Frameworks
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Global Directory Configurations
LOCAL_GRAPH_DIR = "local_project_graphs"
EXPERIMENT_NAME = "classification_fine_tune"

# Set up SQLite backend tracking URI
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment(EXPERIMENT_NAME)


def load_and_clean_data():
    """Loads the dataset, handles cleaning structural verification, and returns dataframes."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # Structural Data Cleaning (Check & handle null rows if present)
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        
    return df, data.feature_names


def generate_and_log_eda_graphs(df):
    """Generates comprehensive exploratory data analysis graphs."""
    eda_dir = os.path.join(LOCAL_GRAPH_DIR, "eda")
    os.makedirs(eda_dir, exist_ok=True)
    
    # 1. Target Class Distribution Visual
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette="viridis")
    plt.title("Target Class Distribution")
    dist_path = os.path.join(eda_dir, "target_distribution.png")
    plt.savefig(dist_path, bbox_inches='tight')
    mlflow.log_artifact(dist_path, "EDA_Graphs")
    plt.close()

    # 2. Correlation Matrix Heatmap (Top 10 features sorted by impact)
    corr = df.corr()
    top_features = corr.nlargest(11, 'target')['target'].index
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[top_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Top 10 Feature Correlation Heatmap")
    corr_path = os.path.join(eda_dir, "correlation_heatmap.png")
    plt.savefig(corr_path, bbox_inches='tight')
    mlflow.log_artifact(corr_path, "EDA_Graphs")
    plt.close()


def generate_and_log_eval_graphs(y_true, y_pred, y_prob, model, trial_num, algo_name, feature_names):
    """Generates and tracks performance valuation visuals per hyperparameter trial run."""
    eval_dir = os.path.join(LOCAL_GRAPH_DIR, "evaluation", f"Trial_{trial_num}_{algo_name}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {algo_name} (Trial {trial_num})")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_path = os.path.join(eval_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    mlflow.log_artifact(cm_path, "Evaluation_Graphs")
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {algo_name}')
    plt.legend(loc="lower right")
    roc_path = os.path.join(eval_dir, "roc_curve.png")
    plt.savefig(roc_path, bbox_inches='tight')
    mlflow.log_artifact(roc_path, "Evaluation_Graphs")
    plt.close()

    # 3. Feature Importance (Applicable for Tree-Based Structural Architectures)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]
        
        plt.figure(figsize=(8, 6))
        plt.title(f"Top 10 Feature Importances: {algo_name}")
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        fi_path = os.path.join(eval_dir, "feature_importance.png")
        plt.savefig(fi_path, bbox_inches='tight')
        mlflow.log_artifact(fi_path, "Evaluation_Graphs")
        plt.close()


# Global Data Setup Context Block
df, feature_names = load_and_clean_data()
X = df.drop('target', axis=1)
y = df['target']

# Feature Engineering: Data Splitting and Feature Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

trial_counter = 0


def objective(space):
    """Conditional Optimization target objective handler across distinct frameworks."""
    global trial_counter
    trial_counter += 1
    
    algo_type = space['type']
    
    with mlflow.start_run(run_name=f"Trial_{trial_counter}_{algo_type}", nested=True):
        
        # Branch instantiation pathways
        if algo_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=int(space['n_estimators']),
                max_depth=int(space['max_depth']) if space['max_depth'] is not None else None,
                min_samples_split=int(space['min_samples_split']),
                random_state=42
            )
            log_module = mlflow.sklearn
            
        elif algo_type == 'xgboost':
            model = XGBClassifier(
                n_estimators=int(space['n_estimators']),
                max_depth=int(space['max_depth']),
                learning_rate=space['learning_rate'],
                subsample=space['subsample'],
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
            log_module = mlflow.xgboost
            
        elif algo_type == 'adaboost':
            model = AdaBoostClassifier(
                n_estimators=int(space['n_estimators']),
                learning_rate=space['learning_rate'],
                random_state=42
            )
            log_module = mlflow.sklearn
            
        elif algo_type == 'ann':
            model = MLPClassifier(
                hidden_layer_sizes=space['hidden_layer_sizes'],
                alpha=space['alpha'],
                learning_rate_init=space['learning_rate_init'],
                max_iter=400,
                random_state=42
            )
            log_module = mlflow.sklearn

        elif algo_type == 'stacked_ensemble':
            # Construct a dynamic meta-architecture stack containing optimized sub-components
            base_estimators = [
                ('rf', RandomForestClassifier(n_estimators=int(space['rf_n_estimators']), max_depth=5, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=50, learning_rate=space['xgb_learning_rate'], use_label_encoder=False, eval_metric='logloss', random_state=42)),
                ('ann', MLPClassifier(alpha=space['ann_alpha'], max_iter=400, random_state=42))
            ]
            model = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(C=space['meta_lr_c']),
                cv=5,
                n_jobs=-1
            )
            log_module = mlflow.sklearn
            
        # Logging Parameters and configurations
        mlflow.log_param("algorithm_family", algo_type)
        mlflow.log_params(space)
        
        # Execute Training Pipeline Phase
        model.fit(X_train_scaled, y_train)
        
        # Calculate evaluation predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc_score": roc_auc_score(y_test, y_prob)
        }
        mlflow.log_metrics(metrics)
        
        # Generate Visual Artifact Folders Locally and inside MLflow Run Core
        generate_and_log_eval_graphs(
            y_true=y_test, 
            y_pred=y_pred, 
            y_prob=y_prob, 
            model=model, 
            trial_num=trial_counter, 
            algo_name=algo_type,
            feature_names=feature_names
        )
        
        # Log Model signature blueprints
        signature = mlflow.models.signature.infer_signature(X_train_scaled, y_pred)
        log_module.log_model(model, "model", signature=signature)
        
        # Hyperopt loss tracking calculation (Minimizing 1 - ROC AUC Score)
        return {'loss': 1 - metrics["auc_score"], 'status': STATUS_OK}


def main():
    # Define multi-algorithm execution space map topology including the Stacked Architecture
    search_space = hp.choice('classifier_type', [
        {
            'type': 'random_forest',
            'n_estimators': hp.quniform('rf_n_estimators', 50, 300, 25),
            'max_depth': hp.choice('rf_max_depth', [3, 5, 10, None]),
            'min_samples_split': hp.choice('rf_min_samples_split', [2, 5, 10])
        },
        {
            'type': 'xgboost',
            'n_estimators': hp.quniform('xgb_n_estimators', 50, 300, 25),
            'max_depth': hp.choice('xgb_max_depth', np.arange(3, 10, dtype=int)),
            'learning_rate': hp.uniform('xgb_learning_rate', 0.01, 0.3),
            'subsample': hp.uniform('xgb_subsample', 0.6, 1.0)
        },
        {
            'type': 'adaboost',
            'n_estimators': hp.quniform('ada_n_estimators', 25, 200, 25),
            'learning_rate': hp.uniform('ada_learning_rate', 0.01, 1.0)
        },
        {
            'type': 'ann',
            'hidden_layer_sizes': hp.choice('ann_layers', [(50,), (100, 50), (64, 32)]),
            'alpha': hp.uniform('ann_alpha', 0.0001, 0.01),
            'learning_rate_init': hp.uniform('ann_lr', 0.001, 0.05)
        },
        {
            'type': 'stacked_ensemble',
            'rf_n_estimators': hp.quniform('stack_rf_n_estimators', 50, 150, 25),
            'xgb_learning_rate': hp.uniform('stack_xgb_lr', 0.01, 0.2),
            'ann_alpha': hp.uniform('stack_ann_alpha', 0.0001, 0.01),
            'meta_lr_c': hp.loguniform('stack_meta_c', np.log(0.01), np.log(10.0))
        }
    ])
    
    print("Initiating Master Cross-Architecture Pipeline Tracking Loop (SQLite Backend)...")
    
    with mlflow.start_run(run_name="Cross_Architecture_Tuning_Parent") as parent_run:
        
        os.makedirs("artifacts", exist_ok=True)
        raw_data_backup = "artifacts/raw_features_dataset.csv"
        df.to_csv(raw_data_backup, index=False)
        mlflow.log_artifact(raw_data_backup, "Dataset_Baseline")
        
        print("Rendering global data EDA profiles...")
        generate_and_log_eda_graphs(df)
        
        trials = Trials()
        
        best_index_profile = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=35,  # Increased evaluations to accommodate the extra stacking space
            trials=trials
        )
        
        print("\nOptimization Sequence Complete.")
        print(f"Optimal Hyperopt array parameter configurations: {best_index_profile}")
        mlflow.log_metric("best_overall_minimum_loss", trials.best_trial['result']['loss'])

    print(f"\nExecution success. Graphs successfully mapped into local directory: './{LOCAL_GRAPH_DIR}'")
    print("Data saved to sqlite:///mlruns.db")

if __name__ == "__main__":
    main()