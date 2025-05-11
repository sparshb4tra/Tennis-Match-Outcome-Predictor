from data_loading import load_data
from data_cleaning import clean_data
from data_preprocessing import preprocess_data
from eda import perform_eda

# FOR XGBOOST
'''from xgboost_model import train_xgboost_model
from model_evaluation import evaluate_model

# Train XGBoost model
print("\nTraining and Evaluating XGBOOST Model...")
xgb_model, X_train, X_test, y_train, y_test, y_pred = train_xgboost_model(df_preprocessed)
print("XGBOOST Model training completed.")

# Evaluate model performance
print("\nEvaluating Model Performance...")
evaluate_model(xgb_model, X_train, X_test, y_train, y_test)'''

# FOR SVM
'''from svm import train_svm_model
from meval_svm import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

print("\nTraining and Evaluating SVM Model...")
svm_model, X_test_svm, y_test_svm, y_pred_svm = train_svm_model(df_preprocessed)
print("SVM Model training completed.")

# Recompute X_train and y_train for SVM evaluation
from svm import prepare_target_and_features
X, y = prepare_target_and_features(df_preprocessed)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)
    
print("\nEvaluating SVM Model Performance...")
evaluate_model(svm_model, X_train_svm, X_test_svm, y_train_svm, y_test_svm)'''

# FOR RF
'''from random_forest_model import train_random_forest_model, prepare_target_and_features
from meval_rf import evaluate_model as evaluate_rf_model
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Train and evaluate Random Forest model
print("\nTraining and Evaluating Random Forest Model...")
rf_model, X_test_rf, y_test_rf, y_pred_rf = train_random_forest_model(df_preprocessed)
print("Random Forest Model training completed.")
    
# Recompute X_train and y_train for Random Forest evaluation
X_rf, y_rf, feature_names = prepare_target_and_features(df_preprocessed)
imputer_rf = SimpleImputer(strategy='mean')
X_imputed_rf = imputer_rf.fit_transform(X_rf)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_imputed_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf
)

print("\nEvaluating Random Forest Model Performance...")
evaluate_rf_model(rf_model, X_train_rf, X_test_rf, y_train_rf, y_test_rf, feature_names)'''

# FOR NB
'''from naive_bayes_model import train_naive_bayes_model, prepare_target_and_features
from meval_nb import evaluate_model as evaluate_nb_model
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Train and evaluate Naive Bayes model
print("\nTraining and Evaluating Naive Bayes Model...")
nb_model, X_test_nb, y_test_nb, y_pred_nb = train_naive_bayes_model(df_preprocessed)
print("Naive Bayes Model training completed.")
    
# Recompute X_train and y_train for Naive Bayes evaluation
X_nb, y_nb = prepare_target_and_features(df_preprocessed)
imputer_nb = SimpleImputer(strategy='mean')
X_imputed_nb = imputer_nb.fit_transform(X_nb)
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
    X_imputed_nb, y_nb, test_size=0.2, random_state=42, stratify=y_nb
)

print("\nEvaluating Naive Bayes Model Performance...")
evaluate_nb_model(nb_model, X_train_nb, X_test_nb, y_train_nb, y_test_nb)'''

# FOR LR
'''from logistic_regression import train_logistic_regression_model, prepare_target_and_features
from meval_lr import evaluate_model as evaluate_lr_model
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Train and evaluate Logistic Regression model
print("\nTraining and Evaluating Logistic Regression Model...")
lr_model, X_test_lr, y_test_lr, y_pred_lr = train_logistic_regression_model(df_preprocessed)
print("Logistic Regression Model training completed.")
    
# Recompute X_train and y_train for Logistic Regression evaluation
X_lr, y_lr = prepare_target_and_features(df_preprocessed)
imputer_lr = SimpleImputer(strategy='mean')
X_imputed_lr = imputer_lr.fit_transform(X_lr)
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_imputed_lr, y_lr, test_size=0.2, random_state=42, stratify=y_lr
)

print("\nEvaluating Logistic Regression Model Performance...")
evaluate_lr_model(lr_model, X_train_lr, X_test_lr, y_train_lr, y_test_lr, X_lr.columns.tolist())'''

# Uncomment the section for the model you want to use
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