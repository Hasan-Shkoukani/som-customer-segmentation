# Customer Segmentation with Self-Organizing Maps

Segments the Mall Customers dataset (200 customers: age, gender, annual income, spending score) with a 10×10 Self-Organizing Map and compares it against a K-Means baseline.

Author: Hasan Shukukani, Near East University, Institute of Graduate Studies

## How it works

1. Drop duplicates and missing rows, encode gender, Min-Max scale all four features.
2. Pick the number of clusters `k` (2–10) from the Elbow, Silhouette and Davies-Bouldin curves. The code takes the `k` with the best Silhouette score.
3. Train a MiniSom (10×10 grid, 10,000 iterations, Gaussian neighborhood, sigma 1.5, learning rate 0.5).
4. Group the SOM's best-matching units into `k` segments with K-Means.
5. Run plain K-Means on the scaled data as the baseline.
6. Report Quantization Error, Topographic Error, Silhouette Score and Davies-Bouldin Index, and save the plots.

## Project structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point. Runs the pipeline and writes the metrics and profile CSVs. |
| `config.py` | Input file, output folder, warning settings. |
| `data.py` | Loading, cleaning and scaling. |
| `models.py` | Choosing `k`, SOM training, SOM labels, K-Means baseline. |
| `plots.py` | All figures. |
| `Mall_Customers.csv` | Dataset ([Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)). |
| `outputs/` | Generated results. |

## Run

```bash
pip install -r requirements.txt
python main.py
```

Run it from the project folder, since paths are relative.

## Outputs (`outputs/`)

| File | Content |
|------|---------|
| `fig1_optimal_k.png` | Elbow, Silhouette and Davies-Bouldin vs. `k` |
| `fig2_umatrix.png` | U-Matrix of the trained SOM |
| `fig3_som_clusters.png` | Samples at their BMU, colored by cluster |
| `fig4_income_spending.png` | Income vs. spending score, SOM and K-Means |
| `fig5_pca.png` | 2-D PCA projection, SOM and K-Means |
| `metrics.csv` | SOM vs. K-Means metrics |
| `cluster_profiles.csv` | Mean age, income and score per SOM cluster |

## References

Kohonen, *Self-Organizing Maps*; MiniSom (Vettigli); Rousseeuw (1987), Davies & Bouldin (1979). Full list in the paper.
# som-customer-segmentation
