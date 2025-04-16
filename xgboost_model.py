# model_training.py        XGBOOST 96%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import numpy as np
from xgboost import XGBClassifier
from numpy import save
def prepare_target_and_features(df):
    # Winner row
    df['target'] = 1

    # Loser row: flip winner and loser features
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

    features = ['rank_diff', 'ace_pct_winner', 'ace_pct_loser',
                'winner_rank', 'loser_rank', 'minutes',
                'surface', 'winner_hand', 'loser_hand', 'round']

    X = df_combined[features]
    y = df_combined['target']
    return X, y, features


def train_xgboost_model(df):
    # Prepare data
    X, y, features = prepare_target_and_features(df)

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # Initialize and train XGBoost
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42,metrics=['aucpr','accuracy'])
    xgb_model.fit(X_train, y_train)

    #Saving the model for future use
    #save.xgb_model()
    #Accuracy and loss plots can be added here for better visualization

    # Predictions (test)
    y_pred = xgb_model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance (XGBoost version)
    importances = xgb_model.feature_importances_
    feature_names = X.columns  # After encoding
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance_df)

    return xgb_model, X_train, X_test, y_train, y_test, y_pred

