import numpy as np
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def choose_optimal_k(X_scaled, k_range=range(2, 11)):
    inertias, silhouettes, dbi_scores = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, lbl))
        dbi_scores.append(davies_bouldin_score(X_scaled, lbl))
    return list(k_range), inertias, silhouettes, dbi_scores


def train_som(X_scaled, grid_size=(10, 10), n_iterations=10000, sigma=1.5, lr=0.5):
    som = MiniSom(
        x=grid_size[0], y=grid_size[1],
        input_len=X_scaled.shape[1],
        sigma=sigma,
        learning_rate=lr,
        neighborhood_function='gaussian',
        random_seed=42
    )
    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, n_iterations, verbose=False)
    return som


def get_som_labels(som, X_scaled, n_clusters):
    bmu_coords = np.array([som.winner(x) for x in X_scaled])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(bmu_coords)
    return labels, bmu_coords


def train_kmeans(X_scaled, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return km.fit_predict(X_scaled), km
