# ========S2=====================================================================
# DAAN: 682: Data Analytics Programming in Python
# Author: Dylan Francis
# Title: Homework_11: Unsupervised learning with Skikit-learn: Clustering 
# =============================================================================
# 1.) Perform data exploratory data analysis and interpret the results. (10 points)
# 2.) Apply K-means clustering to group the seed data. Experiment with different values of k and use unsupervised metrics such as the silhouette score to determine to optimal number of clusters. Compare the results with the ground truth number of clusters. Visualize the clustering result. (30 points)
# 3.) Apply Hierarchical clustering using different linkage methods. Compare the results and determine which linkage method gives the best clustering result. (30 points)
# 4.) Apply DBSCAN clustering to the seed data. Experiment with different values of eps and min_samples and find the combination of eps and min_samples producing the best clustering result based on the number of clusters formed and noise points identified. (30 points)
# 
# =============================================================================
import os
#setting this up to avoid the warnings that come up while operating on windows
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

path = r"C:\Users\dylan\OneDrive\Documents\GRAD_SCHOOL\DAAN_682\HOMEWORK_11"
os.chdir(path)
#setting this up to avoid the warnings that come up while operating on windows
os.environ["OMP_NUM_THREADS"] = "1" 

# Load dataset
seeds_dataset = pd.read_csv("seeds_dataset.csv", header=None)
#assign column headers since the data didn't have any
columns = ["area", "perimeter", "compactness", "length_kernel",
           "width_kernel", "asymmetry_coefficient", "groove_length", "label"]
seeds_dataset.columns = columns

# breaking up the data
x = seeds_dataset.iloc[:, :-1]
y_true = seeds_dataset["label"]

# 1.) Data Exploration
print(f"\nThere are {seeds_dataset.shape[0]} rows, and {seeds_dataset.shape[1]} columns.")
print("\n======= DATASET INFO ======")
print(x.info())
print("\n======= SUMMARY STATS ======")
print(x.describe())
print("\n======= CORRELATION MATRIX ======")
corr_matrix = x.corr()
print(corr_matrix)
print("\n===== MISSING VALUES =====")
print(x.isnull().sum())
print("\n===== DUPLICATE ROWS =====")
print(f"The total number of duplicate rows is: {x.duplicated().sum()}")

# Histograms
x.hist(figsize=(10, 8), bins=15)
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=x)
plt.title("Boxplot of All Features")
plt.xticks(rotation=45)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(x)
plt.suptitle("Pairwise Feature Relationships", y=1.02)
plt.show()
# =============================================================================

#2. K means 

#Scaling the data / normalize
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 3. Find Optimal K using Silhouette Score
k_values = range(2, 10)
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=13, n_init=10)
    labels = kmeans.fit_predict(x_scaled)
    score = silhouette_score(x_scaled, labels)
    silhouette_scores.append(score)

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='x')
plt.title("Silhouette Score vs. Number of Clusters (k)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid()
plt.show()

# determining best k
optimal_k = k_values[np.argmax(silhouette_scores)]
print(f"Optimal k based on silhouette score: {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=13, n_init=10)
clusters = kmeans.fit_predict(x_scaled)

# Compare with Truth
ari = adjusted_rand_score(y_true, clusters)
print(f"Adjusted Rand Index (comparison with true labels): {ari:.3f}")

print(f"True number of clusters: {len(np.unique(y_true))}")
print(f"KMeans clusters: {optimal_k}")
if optimal_k != len(np.unique(y_true)):
    print("The k mean model is not yielding expected number of clusters")

# Visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(12, 5))

# Plot KMeans clusters
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title(f"KMeans Clusters (k={optimal_k})")

# Plot ground truth
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis')
plt.title("Ground Truth Labels")
plt.show()

# =============================================================================
# 3.) Hierarchical clustering
linkage_methods = ["ward", "average", "complete"]
results = {}

for linkage in linkage_methods:
    model = AgglomerativeClustering(n_clusters=3, metric = 'euclidean', linkage=linkage)
    labels = model.fit_predict(x_scaled)

    sil_score = silhouette_score(x_scaled, labels)
    ari = adjusted_rand_score(y_true, labels)

    results[linkage] = {
        "silhouette": sil_score,
        "ari": ari,
        "labels": labels
    }
print("Linkage Method Comparison:\n")
for method, metrics in results.items():
    print(f"{method.upper()} | Silhouette: {metrics['silhouette']:.3f} | ARI: {metrics['ari']:.3f}")


# Find Best Linkage
best_method_sil = max(results, key=lambda x: results[x]["silhouette"])
best_method_ari = max(results, key=lambda x: results[x]["ari"])

if best_method_sil == best_method_ari:
    print(f"\nBest linkage method (by silhouette and AIR score) is: {best_method_sil.upper()}")


# Visualization through PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(12, 8))

for i, method in enumerate(linkage_methods, 1):
    plt.subplot(2, 2, i)
    plt.scatter(X_pca[:, 0], X_pca[:, 1],
                c=results[method]["labels"],
                cmap="viridis")
    plt.title(f"{method.upper()} Linkage")

plt.tight_layout()
plt.show()
# =============================================================================
# 3.) DBScan Clustering

db_result = []

epses = np.arange(0.7, 1.1, 0.05) 
min_samples_list = [3, 5, 8, 10, 12, 15]

for eps in epses:
    for min_samples in min_samples_list:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(x_scaled)

        # Number of clusters (excluding noise)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Number of noise points
        n_noise = list(labels).count(-1)

        # Silhouette score (only valid if >1 cluster)
        if n_clusters > 1:
            score = silhouette_score(x_scaled, labels)
        else:
            score = -1  # invalid case

        db_result.append({
            "eps": eps,
            "min_samples": min_samples,
            "clusters": n_clusters,
            "noise_points": n_noise,
            "silhouette": score
        })

# Convert to DataFrame 
df_results = pd.DataFrame(db_result)

# Sort results: prioritize high silhouette, more clusters, fewer noise points
df_results = df_results.sort_values(
    by=["clusters", "noise_points", "silhouette",],
    ascending=[False, True, False]
)

print(df_results)

print("A esp of 1.0 and sample size of 12 produces an auurate estaimte of the clusters 3, and the lowest noisepoints of 55")