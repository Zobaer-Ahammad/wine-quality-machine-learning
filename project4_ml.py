"""CSCI 5260 Project 4 - Part 2: Supervised wine-quality classification.

Models: Decision Tree and K-Nearest Neighbors. Both are evaluated on THE SAME
stratified 80/20 train/test split as Part 3 (seed=42). The quality column is
predictor target; type 0=red and 1=white is an input feature.
Run: python project4_ml.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score,
                             multilabel_confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results'
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_split():
    """Make the exact same split as Part 3 to enable fair comparison."""
    red = pd.read_csv(BASE / 'winequality-red.csv')
    white = pd.read_csv(BASE / 'winequality-white.csv')
    red['type'] = 0
    white['type'] = 1
    combined = pd.concat([red, white], ignore_index=True)
    target = combined['quality'].copy()
    features = combined.drop(columns=['quality'])
    return train_test_split(features, target, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=target)


def calculate_metrics(y_true, y_pred, labels, model_name: str) -> dict:
    """Compute accuracy and TP/TN/FP/FN for each quality in a multiclass task.

    For each quality class c, 'positive' means quality=c, and 'negative' means
    all other quality classes. This is the one-versus-rest interpretation.
    """
    matrices = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    per_class = []
    for quality, values in zip(labels, matrices):
        tn, fp, fn, tp = values.ravel()
        per_class.append({'quality': int(quality), 'TN': int(tn),
                          'FP': int(fp), 'FN': int(fn), 'TP': int(tp)})
    pd.DataFrame(per_class).to_csv(OUT / f'{model_name}_per_class.csv', index=False)

    print(f'\n{model_name.upper()}')
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print(pd.DataFrame(per_class).to_string(index=False))
    print(f'Accuracy: {accuracy_score(y_true, y_pred):.4f}; '
          f'macro F1: {f1_score(y_true, y_pred, average="macro", zero_division=0):.4f}')

    fig, ax = plt.subplots(figsize=(6.1, 5.2))
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=labels)
    display.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title(model_name.replace('_', ' ').title() + ': test-set confusion matrix')
    fig.tight_layout()
    fig.savefig(OUT / f'{model_name}_confusion.png', dpi=170)
    plt.close(fig)

    return {
        'model_name': model_name, 'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'class_labels': [int(i) for i in labels],
        'confusion_matrix': confusion.tolist(),
        'per_class': per_class,
        'sample_count_test': int(len(y_true))
    }


def main():
    X_train, X_test, y_train, y_test = load_split()
    labels = np.sort(pd.concat([y_train, y_test]).unique())
    print(f'Samples: train={len(X_train)}, test={len(X_test)}, '
          f'features={X_train.shape[1]}, classes={labels.tolist()}')
    print(f'Train class counts: {y_train.value_counts().sort_index().to_dict()}')
    print(f'Test class counts: {y_test.value_counts().sort_index().to_dict()}')

    # Tree decisions are based on feature thresholds; standardization not needed.
    tree = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5,
                                  random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    tree_predictions = tree.predict(X_test)
    tree_metrics = calculate_metrics(y_test, tree_predictions, labels, 'decision_tree')

    # KNN uses distances, so scale training inputs and apply the SAME transform to test.
    # The Pipeline prevents scaling information leaking from the test set.
    knn = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=9))
    ])
    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    knn_metrics = calculate_metrics(y_test, knn_predictions, labels, 'knn')

    result = {'setup': {'test_size': TEST_SIZE, 'random_state': RANDOM_STATE,
                         'split': 'stratified on quality',
                         'train_n': int(len(y_train)), 'test_n': int(len(y_test)),
                         'target': 'quality', 'type_coding': 'red=0, white=1',
                         'classes': labels.tolist(),
                         'train_counts': {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
                         'test_counts': {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()}},
              'models': {'decision_tree': tree_metrics, 'knn': knn_metrics}}
    (OUT / 'ml_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('\nSaved per-class counts, confusion matrices, and results JSON to results/.')


if __name__ == '__main__':
    main()
