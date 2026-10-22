import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load the local dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Hierarchical clustering
hierarchical = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Evaluate clustering algorithms
results = []

for name, labels in [
    ("K-Means", kmeans_labels),
    ("Hierarchical Clustering", hierarchical_labels)
]:
    results.append({
        "Algorithm": name,
        "Silhouette Score": round(
            silhouette_score(X_scaled, labels), 4
        ),
        "Davies-Bouldin Index": round(
            davies_bouldin_score(X_scaled, labels), 4
        )
    })

results_df = pd.DataFrame(results)

print("\nClustering Evaluation Results")
print(results_df.to_string(index=False))

# Generate cluster graphs
plots = [
    ("K-Means Clustering", kmeans_labels, "kmeans_clusters.png"),
    (
        "Hierarchical Clustering",
        hierarchical_labels,
        "hierarchical_clusters.png"
    )
]

for title, labels, filename in plots:
    plt.figure(figsize=(8, 5))

    plt.scatter(
        X["Annual Income (k$)"],
        X["Spending Score (1-100)"],
        c=labels,
        cmap="viridis",
        s=50
    )

    plt.title(title)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()

# Save evaluation metrics
results_df.to_csv("clustering_metrics.csv", index=False)

print("\nGraphs and evaluation results saved successfully.")