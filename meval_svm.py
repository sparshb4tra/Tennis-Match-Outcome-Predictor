import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
import numpy as np

def evaluate_model(svm_model, X_train, X_test, y_train, y_test):
    # Predictions
    y_train_pred = svm_model.predict(X_train)
    y_test_pred = svm_model.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Loss", "Win"], yticklabels=["Loss", "Win"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # Feature Importance (for linear SVM)
    if svm_model.kernel == 'linear':
        feature_importance = np.abs(svm_model.coef_[0])
        feature_names = X_train.shape[1]  # Assuming X_train is numpy array
        indices = np.argsort(feature_importance)[::-1]
        
        print("\nTop 5 Most Important Features:")
        for i in range(min(5, len(feature_importance))):
            print(f"Feature {indices[i]}: {feature_importance[indices[i]]:.4f}")

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importance)), feature_importance[indices])
        plt.title("Feature Importance (Absolute SVM Coefficients)")
        plt.xlabel("Feature Index")
        plt.ylabel("Absolute Coefficient Value")
        plt.savefig('feature_importance.png')
        plt.close()