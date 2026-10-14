import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

from config import OUTPUT_DIR
from data import load_and_preprocess
from models import choose_optimal_k, train_som, get_som_labels, train_kmeans
from plots import (plot_optimal_k, plot_umatrix, plot_som_clusters,
                   plot_income_spending, plot_pca)


def main():
    print("Starting data preprocessing...")
    df, X_scaled, features, scaler = load_and_preprocess()
    print(f"✓ Data loaded: {df.shape[0]} customers")

    print("Finding optimal K...")
    k_values, inertias, silhouettes, dbi_scores = choose_optimal_k(X_scaled, k_range=range(2, 11))
    plot_optimal_k(k_values, inertias, silhouettes, dbi_scores)
    N_CLUSTERS = int(np.argmax(silhouettes)) + 2
    print(f"✓ Optimal K: {N_CLUSTERS}")

    print("Training SOM...")
    som = train_som(X_scaled)
    qe = som.quantization_error(X_scaled)
    te = som.topographic_error(X_scaled)
    print(f"✓ SOM trained - QE: {qe:.6f}, TE: {te:.6f}")

    print("Getting SOM labels...")
    som_labels, _ = get_som_labels(som, X_scaled, N_CLUSTERS)
    som_sil = silhouette_score(X_scaled, som_labels)
    som_dbi = davies_bouldin_score(X_scaled, som_labels)

    print("Training K-Means...")
    km_labels, _ = train_kmeans(X_scaled, N_CLUSTERS)
    km_sil = silhouette_score(X_scaled, km_labels)
    km_dbi = davies_bouldin_score(X_scaled, km_labels)

    print(f"\n--- Results ---")
    print(f"SOM  - Silhouette: {som_sil:.4f}, DBI: {som_dbi:.4f}")
    print(f"KM   - Silhouette: {km_sil:.4f}, DBI: {km_dbi:.4f}")

    metrics = {
        'Quantization Error': [qe, None],
        'Topographic Error': [te, None],
        'Silhouette Score': [som_sil, km_sil],
        'Davies-Bouldin Index': [som_dbi, km_dbi],
        'Optimal K': [N_CLUSTERS, N_CLUSTERS],
    }
    pd.DataFrame(metrics, index=['SOM', 'K-Means']).T.to_csv(f"{OUTPUT_DIR}/metrics.csv")

    df_out = df.copy()
    df_out['SOM_Cluster'] = som_labels
    cluster_means = df_out.groupby('SOM_Cluster')[
        ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
    cluster_means.to_csv(f"{OUTPUT_DIR}/cluster_profiles.csv")

    print("Generating plots...")
    plot_umatrix(som)
    plot_som_clusters(som, X_scaled, som_labels)
    plot_income_spending(df, som_labels, km_labels)
    plot_pca(X_scaled, som_labels, km_labels)

    print(f"\n✓ All outputs saved to ./{OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
