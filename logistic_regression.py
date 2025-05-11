# logistic_regression.py  65%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import numpy as np

def prepare_target_and_features(df):
    # Winner row
    df['target'] = 1

    # Loser row: flip relevant features
    df_loser = df.copy()
    df_loser['target'] = 0

    # Flip features
    df_loser = df_loser.rename(columns={
        'winner_rank': 'loser_rank', 'loser_rank': 'winner_rank',
        'ace_pct_winner': 'ace_pct_loser', 'ace_pct_loser': 'ace_pct_winner',
        'winner_hand': 'loser_hand', 'loser_hand': 'winner_hand'
    })

    # Combine both
    df_combined = pd.concat([df, df_loser]).reset_index(drop=True)

    # Select features
    features = ['rank_diff', 'ace_pct_winner', 'ace_pct_loser', 'winner_rank', 
                'loser_rank', 'minutes', 'surface', 'winner_hand', 'loser_hand', 'round']
    
    X = df_combined[features]
    y = df_combined['target']
    
    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    return X, y

def train_logistic_regression_model(df):
    # Prepare data
    X, y = prepare_target_and_features(df)

    # Impute missing values in X
    imputer = SimpleImputer(strategy='mean')  # Use mean to replace NaN
    X_imputed = imputer.fit_transform(X)

    # Divide the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # Initialize and train the Logistic Regression model
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)

    # Make predictions
    y_pred = lr_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return lr_model, X_test, y_test, y_pred

# Example usage (integrate this into main.py)
if __name__ == "__main__":
    # Assuming df_preprocessed is available
    df = pd.read_csv('atp_matches_2024.csv')  # Placeholder; replace with your loading logic
    from data_cleaning import clean_data
    from data_preprocessing import preprocess_data
    df_cleaned = clean_data(df)
    df_preprocessed = preprocess_data(df_cleaned)
    model, X_test, y_test, y_pred = train_logistic_regression_model(df_preprocessed)