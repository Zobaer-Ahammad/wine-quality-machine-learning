"""CSCI 5260 Project 4 - Part 3: MLP neural-network classifier.

Uses the same 80/20 split, ordering, random seed, and quality stratification as
project4_ml.py. The scaler is fit to TRAINING data only (no test leakage).
Run: python project4_nn.py
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results'
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_split():
    red = pd.read_csv(BASE / 'winequality-red.csv')
    white = pd.read_csv(BASE / 'winequality-white.csv')
    red['type'] = 0
    white['type'] = 1
    combined = pd.concat([red, white], ignore_index=True)
    target = combined['quality'].copy()
    features = combined.drop(columns=['quality'])
    return train_test_split(features, target, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=target)


def main():
    X_train, X_test, y_train, y_test = load_split()
    labels = np.sort(pd.concat([y_train, y_test]).unique())
    print(f'Samples: train={len(X_train)}, test={len(X_test)}, '
          f'features={X_train.shape[1]}')

    # The neural network receives scaled inputs, but scaling parameters come
    # only from the training set. The hidden layers have 64 and 32 neurons.
    neural_net = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=450,
                              early_stopping=True, n_iter_no_change=20,
                              random_state=RANDOM_STATE))
    ])
    neural_net.fit(X_train, y_train)
    predictions = neural_net.predict(X_test)

    confusion = confusion_matrix(y_test, predictions, labels=labels)
    matrices = multilabel_confusion_matrix(y_test, predictions, labels=labels)
    per_class = []
    for quality, values in zip(labels, matrices):
        tn, fp, fn, tp = values.ravel()
        per_class.append({'quality': int(quality), 'TN': int(tn),
                          'FP': int(fp), 'FN': int(fn), 'TP': int(tp)})
    pd.DataFrame(per_class).to_csv(OUT / 'mlp_per_class.csv', index=False)
    print('\nMLP NEURAL NETWORK')
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))
    print(pd.DataFrame(per_class).to_string(index=False))
    print(f'Accuracy: {accuracy_score(y_test, predictions):.4f}; '
          f'macro F1: {f1_score(y_test, predictions, average="macro", zero_division=0):.4f}')

    fig, ax = plt.subplots(figsize=(6.1, 5.2))
    ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=labels).plot(
        ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title('MLP neural network: test-set confusion matrix')
    fig.tight_layout()
    fig.savefig(OUT / 'mlp_confusion.png', dpi=170)
    plt.close(fig)

    mlp = neural_net.named_steps['mlp']
    result = {
        'model_name': 'mlp', 'train_n': int(len(y_train)), 'test_n': int(len(y_test)),
        'test_size': TEST_SIZE, 'random_state': RANDOM_STATE,
        'class_labels': [int(i) for i in labels],
        'accuracy': float(accuracy_score(y_test, predictions)),
        'macro_f1': float(f1_score(y_test, predictions, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, predictions)),
        'iterations': int(mlp.n_iter_), 'hidden_layer_sizes': [64, 32],
        'confusion_matrix': confusion.tolist(), 'per_class': per_class
    }
    (OUT / 'nn_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Finished MLP training in {mlp.n_iter_} iterations.')
    print('\nSaved neural-network results to results/.')


if __name__ == '__main__':
    main()
