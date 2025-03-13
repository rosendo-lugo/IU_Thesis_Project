'''
*------------------*
|                  |
|     MODELING     |
|                  |
*------------------*
'''

 # ----------------------------------------------------------------------------------  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ---------------------------------------------------------------------------------- 
def preprocess_data(df, target):
    """
    Encodes categorical variables and prepares features/target for modeling.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  

    X = df.drop(columns=[target])
    y = df[target]

    return X, y, label_encoders


# ----------------------------------------------------------------------------------
# Split function
def get_split(X, y, test_size=0.2, val_size=0.25, random_state=123):
    """
    Splits the dataset into training, validation, and test sets.

    - Training: 60%
    - Validation: 20%
    - Test: 20%
    """
    X_train_validate, X_test, y_train_validate, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_validate, y_train_validate, test_size=val_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test
    
# ---------------------------------------------------------------------------------- 
def calculate_baseline(y_train):
    """
    Computes the baseline for regression tasks using the mean of y_train.
    """
    return round(y_train.mean(), 2)  # We use the mean as a baseline predictor

# ---------------------------------------------------------------------------------- 
def calculate_mape(y_train):
    """
    Computes the Mean Absolute Percentage Error (MAPE) using the mean sale price.
    """
    baseline_pred = y_train.mean()
    mape = np.mean(np.abs((y_train - baseline_pred) / y_train)) * 100
    return round(mape, 2)
    
# ---------------------------------------------------------------------------------- 
def summarize_split(X_train, X_val, X_test, baseline_value):
    """
    Creates a summary of the dataset split and baseline (mean sale price).
    """
    split_summary = pd.DataFrame({
        "Dataset": ["Training", "Validation", "Test"],
        "Size": [X_train.shape[0], X_val.shape[0], X_test.shape[0]],
        "Feature Count": [X_train.shape[1]] * 3,
        "Baseline (Mean Sale Price)": [baseline_value] * 3
    })
    return split_summary
    
# ---------------------------------------------------------------------------------- 
def calculate_mape(y_train):
    """
    Computes the Mean Absolute Percentage Error (MAPE) using the mean sale price.
    """
    baseline_pred = y_train.mean()  # Predicting the mean as baseline
    mape = np.mean(np.abs((y_train - baseline_pred) / y_train)) * 100  # Convert to %
    return round(mape, 2)

# ---------------------------------------------------------------------------------- 
def get_classification_models():
    """
    Returns a dictionary of classification models.
    """
    return {
        "Naïve Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Zero-R (Baseline)": DummyClassifier(strategy="most_frequent"),
    }
# ---------------------------------------------------------------------------------- 
def get_regression_models():
    """
    Returns a dictionary of regression models.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
# ---------------------------------------------------------------------------------- 
def evaluate_classification(y_test, y_pred):
    """
    Computes evaluation metrics for classification models.
    """
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
# ---------------------------------------------------------------------------------- 
def evaluate_regression(y_test, y_pred):
    """
    Computes evaluation metrics for regression models.
    """
    return {
        "MSE": mean_squared_error(y_test, y_pred),
        "R2 Score": r2_score(y_test, y_pred)
    }

# ---------------------------------------------------------------------------------- 
def train_and_evaluate_classification(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates classification models.
    """
    models = get_classification_models()
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_classification(y_test, y_pred)
        metrics["Model"] = name  # Add model name to dictionary
        results.append(metrics)

    # Convert to DataFrame, drop unnecessary columns, and reorder
    classification_results = pd.DataFrame(results).drop(columns=["MSE", "R2 Score"], errors="ignore")
    classification_results = classification_results[["Model"] + [col for col in classification_results.columns if col != "Model"]]
    return classification_results
    
# ---------------------------------------------------------------------------------- 
def train_and_evaluate_regression(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates regression models.
    """
    models = get_regression_models()
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_regression(y_test, y_pred)
        metrics["Model"] = name  # Add model name to dictionary
        results.append(metrics)

    # Convert to DataFrame, drop unnecessary columns, and reorder
    regression_results = pd.DataFrame(results).drop(columns=["Accuracy", "Precision", "Recall", "F1-Score"], errors="ignore")
    regression_results = regression_results[["Model"] + [col for col in regression_results.columns if col != "Model"]]
    return regression_results
# ----------------------------------------------------------------------------------
# Feature selection
def feature_selection(X_train, y_train, n_features=15):
    """
    Performs Recursive Feature Elimination (RFE) to select important features.
    """
    base_model = GradientBoostingRegressor()
    rfe = RFE(base_model, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)

    selected_features = X_train.columns[rfe.support_]
    return selected_features

# ----------------------------------------------------------------------------------
# Train models
def train_models(X_train, y_train, models):
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models

# ----------------------------------------------------------------------------------
# Evaluate models
def evaluate_models(trained_models, X_test, y_test):
    results = []

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({"Model": name, "MSE": mse, "R² Score": r2})

    return pd.DataFrame(results)

# ----------------------------------------------------------------------------------
# Hyperparameter tuning
def hyperparameter_tuning(X_train, y_train, model_name):
    """
    Tunes hyperparameters using GridSearchCV.
    """
    param_grid = {}

    if model_name == "GradientBoosting":
        model = GradientBoostingRegressor()
        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        }
    elif model_name == "RandomForest":
        model = RandomForestRegressor()
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10],
        }
    elif model_name == "XGBoost":
        model = XGBRegressor()
        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        }
    elif model_name == "LightGBM":
        model = LGBMRegressor()
        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31, 40],
        }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_
# ---------------------------------------------------------------------------------- 
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains multiple models and evaluates them using appropriate metrics.
    """
    models = {
        "Naïve Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Zero-R (Baseline)": DummyClassifier(strategy="most_frequent"),
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if isinstance(model, (LinearRegression, RandomForestRegressor, GradientBoostingRegressor, DecisionTreeRegressor)):
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results.append({
                "Model": name,
                "MSE": mse,
                "R2 Score": r2,
                "Accuracy": None,
                "Precision": None,
                "Recall": None,
                "F1-Score": None
            })
        else:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            results.append({
                "Model": name,
                "MSE": None,
                "R2 Score": None,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            })

    return pd.DataFrame(results)

# ---------------------------------------------------------------------------------- 



