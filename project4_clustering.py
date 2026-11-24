"""CSCI 5260 Project 4 - Part 1: K-Means clustering of red and white wines.

Run: python project4_clustering.py
The 11 chemical input columns are standardized before clustering. The quality
column is held out and used only AFTER clustering to evaluate each group.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')  # Save figures instead of requiring a desktop window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results'
OUT.mkdir(exist_ok=True)


def summarize_clusters(wine_name: str, data: pd.DataFrame, target: pd.Series) -> dict:
    """Cluster one wine feature table and summarize its held-out true quality."""

    # The attributes have very different numerical ranges, so standardize them.
    # StandardScaler is learned from this one unsupervised dataset only.
    features = StandardScaler().fit_transform(data)
    model = KMeans(n_clusters=11, n_init=20, random_state=42)
    cluster_labels = model.fit_predict(features)

    # Important: 0..10 below are arbitrary cluster IDs, NOT quality predictions.
    cluster_table = pd.DataFrame({'cluster': cluster_labels, 'quality': target})
    summary = (cluster_table.groupby('cluster')['quality']
               .agg(count='size', mean='mean', median='median', std='std',
                    lowest='min', highest='max')
               .sort_index())
    summary.to_csv(OUT / f'{wine_name}_cluster_summary.csv', float_format='%.4f')

    # Figure 1 for this wine: mean true wine-quality rating in each cluster.
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cluster_ids = summary.index.to_numpy()
    ax.bar(cluster_ids.astype(str), summary['mean'], color='#326999')
    for j, (cluster_id, row) in enumerate(summary.iterrows()):
        ax.text(j, row['mean'] + 0.15, f"{row['mean']:.2f}", ha='center',
                va='bottom', fontsize=8)
    ax.set(title=f'{wine_name.title()} wine: mean real quality by K-Means cluster',
           xlabel='K-Means cluster ID (arbitrary group)',
           ylabel='Mean actual wine quality (0–10)')
    ax.set_ylim(0, 10)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / f'{wine_name}_cluster_mean.png', dpi=170)
    plt.close(fig)

    # Figure 2: distribution across wines in each cluster (not just averages).
    grouped_values = [cluster_table.loc[cluster_table.cluster == k, 'quality']
                      for k in cluster_ids]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.boxplot(grouped_values, tick_labels=[str(k) for k in cluster_ids],
               showfliers=True, patch_artist=True,
               boxprops={'facecolor': '#d7e5ef', 'edgecolor': '#326999'},
               medianprops={'color': '#b94040', 'linewidth': 2})
    ax.set(title=f'{wine_name.title()} wine: actual quality distribution in each cluster',
           xlabel='K-Means cluster ID (arbitrary group)',
           ylabel='Actual quality rating (0–10)')
    ax.set_ylim(0, 10)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / f'{wine_name}_cluster_distribution.png', dpi=170)
    plt.close(fig)

    print(f'\n{wine_name.upper()} ({len(data)} wines, {len(data.columns)} input features)')
    print(f'K-Means: clusters=11; iterations={model.n_iter_}; inertia={model.inertia_:.2f}')
    print(summary.to_string(float_format=lambda value: f'{value:.2f}'))
    return {
        'wine_type': wine_name,
        'samples': int(len(data)), 'feature_count': int(len(data.columns)),
        'n_clusters': 11, 'iterations': int(model.n_iter_),
        'inertia': float(model.inertia_),
        'actual_quality_values': sorted(target.unique().astype(int).tolist()),
        'cluster_quality_range': [float(summary['mean'].min()),
                                  float(summary['mean'].max())],
        'clusters': [
            {'cluster': int(cluster_id),
             'count': int(r['count']), 'mean': float(r['mean']),
             'median': float(r['median']), 'std': float(r['std'])
              if pd.notna(r['std']) else None,
             'lowest': int(r['lowest']), 'highest': int(r['highest'])}
            for cluster_id, r in summary.iterrows()
        ]
    }


def main():
    # Follow assignment Steps 1 and 2 explicitly: two data frames and targets.
    data_white = pd.read_csv(BASE / 'winequality-white.csv')
    data_red = pd.read_csv(BASE / 'winequality-red.csv')
    target_white = data_white['quality'].copy()
    target_red = data_red['quality'].copy()
    data_white = data_white.drop(columns=['quality'])
    data_red = data_red.drop(columns=['quality'])

    # Only chemical input variables enter the unsupervised clustering step.
    white = summarize_clusters('white', data_white, target_white)
    red = summarize_clusters('red', data_red, target_red)
    (OUT / 'clustering_summary.json').write_text(
        json.dumps({'white': white, 'red': red}, indent=2), encoding='utf-8')
    print('\nSaved cluster tables, 4 plots, and summary JSON to results/.')


if __name__ == '__main__':
    main()
