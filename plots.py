import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

from config import OUTPUT_DIR


def plot_optimal_k(k_range, inertias, silhouettes, dbi_scores):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(list(k_range), inertias, 'bo-')
    axes[0].set_title('Elbow Method')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia')
    axes[0].grid(True)

    axes[1].plot(list(k_range), silhouettes, 'gs-')
    axes[1].set_title('Silhouette Score vs K')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].grid(True)

    axes[2].plot(list(k_range), dbi_scores, 'r^-')
    axes[2].set_title('Davies-Bouldin Index vs K')
    axes[2].set_xlabel('Number of Clusters (K)')
    axes[2].set_ylabel('DB Index')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_optimal_k.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_umatrix(som):
    umatrix = som.distance_map()
    plt.figure(figsize=(7, 6))
    plt.pcolor(umatrix.T, cmap='bone_r')
    plt.colorbar(label='Distance')
    plt.title('U-Matrix')
    plt.xlabel('SOM x')
    plt.ylabel('SOM y')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_umatrix.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_som_clusters(som, X_scaled, labels):
    umatrix = som.distance_map()
    n_clusters = len(np.unique(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    plt.figure(figsize=(7, 6))
    plt.pcolor(umatrix.T, cmap='bone_r', alpha=0.6)
    plt.colorbar(label='Distance')
    for i, x in enumerate(X_scaled):
        bmu = som.winner(x)
        plt.plot(bmu[0] + 0.5, bmu[1] + 0.5, 'o',
                 color=colors[labels[i]], markersize=5, alpha=0.7)
    patches = [mpatches.Patch(color=colors[c], label=f'Cluster {c}')
               for c in np.unique(labels)]
    plt.legend(handles=patches, loc='upper right', fontsize=8)
    plt.title('SOM Cluster Mapping')
    plt.xlabel('SOM x')
    plt.ylabel('SOM y')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3_som_clusters.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_income_spending(df, som_labels, km_labels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc1 = axes[0].scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                          c=som_labels, cmap='tab10', s=40, alpha=0.8)
    axes[0].set_title('SOM – Income vs Spending Score')
    axes[0].set_xlabel('Annual Income (k$)')
    axes[0].set_ylabel('Spending Score (1-100)')
    plt.colorbar(sc1, ax=axes[0], label='Cluster')

    sc2 = axes[1].scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                          c=km_labels, cmap='tab10', s=40, alpha=0.8)
    axes[1].set_title('K-Means – Income vs Spending Score')
    axes[1].set_xlabel('Annual Income (k$)')
    axes[1].set_ylabel('Spending Score (1-100)')
    plt.colorbar(sc2, ax=axes[1], label='Cluster')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_income_spending.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_pca(X_scaled, som_labels, km_labels):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=som_labels,
                          cmap='tab10', s=30, alpha=0.8)
    axes[0].set_title('SOM Clusters – PCA Projection')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    plt.colorbar(sc1, ax=axes[0], label='Cluster')

    sc2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels,
                          cmap='tab10', s=30, alpha=0.8)
    axes[1].set_title('K-Means Clusters – PCA Projection')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    plt.colorbar(sc2, ax=axes[1], label='Cluster')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_pca.png", dpi=150, bbox_inches='tight')
    plt.close()
