from data_loading import load_data
from data_cleaning import clean_data
from data_preprocessing import preprocess_data
from eda import perform_eda
from xgboost_model import train_xgboost_model
from model_evaluation import evaluate_model  # NEW IMPORT

# Define your CSV file path
file_path = 'atp_matches_2024.csv'

# Load the data
df = load_data(file_path)

# Clean the data
df_cleaned = clean_data(df)

# Preprocess the data
df_preprocessed = preprocess_data(df_cleaned)

# Perform EDA
perform_eda(df_preprocessed)

# Train XGBoost model
print("\nTraining and Evaluating XGBOOST Model...")
xgb_model, X_train, X_test, y_train, y_test, y_pred = train_xgboost_model(df_preprocessed)
print("XGBOOST Model training completed.")

# Evaluate model performance
print("\nEvaluating Model Performance...")
evaluate_model(xgb_model, X_train, X_test, y_train, y_test)
