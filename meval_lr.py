import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score,
    roc_curve, confusion_matrix, classification_report
)

def evaluate_model(lr_model, X_train, X_test, y_train, y_test, feature_names=None):
    # Predictions
    y_train_pred = lr_model.predict(X_train)
    y_test_pred = lr_model.predict(X_test)

    # Probabilities for ROC and Log Loss
    y_train_proba = lr_model.predict_proba(X_train)[:, 1]
    y_test_proba = lr_model.predict_proba(X_test)[:, 1]

    # Accuracy and Log Loss
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_loss = log_loss(y_train, y_train_proba)
    test_loss = log_loss(y_test, y_test_proba)

    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    print(f"Training Log Loss: {train_loss:.4f}")
    print(f"Testing Log Loss: {test_loss:.4f}")

    # ROC-AUC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="navy")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve")
    plt.legend()
    plt.savefig('roc_curve_lr.png')
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Loss", "Win"], yticklabels=["Loss", "Win"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig('confusion_matrix_lr.png')
    plt.close()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # Feature Importance (based on absolute coefficients)
    if feature_names is not None:
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': np.abs(lr_model.coef_[0])
        }).sort_values('coefficient', ascending=False)
        
        print("\nTop 5 Most Important Features (Absolute Coefficients):")
        print(feature_importance.head())

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='coefficient', y='feature', data=feature_importance.head(10))
        plt.title("Top 10 Feature Importance (Absolute Coefficients)")
        plt.xlabel("Absolute Coefficient Value")
        plt.ylabel("Feature")
        plt.savefig('feature_importance_lr.png')
        plt.close()