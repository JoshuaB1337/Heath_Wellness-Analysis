import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import plotly.graph_objects as go

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic health and wellness data
n_samples = 200

# Generate features
exercise_time = np.random.normal(45, 15, n_samples)  # Daily exercise time in minutes
healthy_meals = np.random.normal(2.5, 0.8, n_samples)  # Number of healthy meals per day
sleep_hours = np.random.normal(7, 1, n_samples)  # Hours of sleep per night
stress_level = np.random.normal(5, 2, n_samples)  # Stress level (1-10)
bmi = np.random.normal(25, 4, n_samples)  # BMI

# Ensure values are within reasonable ranges
exercise_time = np.clip(exercise_time, 0, 120)
healthy_meals = np.clip(healthy_meals, 0, 5)
sleep_hours = np.clip(sleep_hours, 4, 12)
stress_level = np.clip(stress_level, 1, 10)
bmi = np.clip(bmi, 15, 40)

# Create DataFrame
df = pd.DataFrame({
    'exercise_time': exercise_time,
    'healthy_meals': healthy_meals,
    'sleep_hours': sleep_hours,
    'stress_level': stress_level,
    'bmi': bmi
})

# Save the dataset
df.to_csv('health_wellness_data.csv', index=False)

# Display basic information about the dataset
print("\nDataset Overview:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# Create correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Health and Wellness Indicators')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Function to evaluate clustering
def evaluate_clustering(X, labels):
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    return silhouette, calinski

# Perform K-means clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Find optimal number of clusters using elbow method
inertias = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.savefig('elbow_curve.png')
plt.close()

# Perform K-means with optimal clusters (let's use 3)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Evaluate K-means clustering
silhouette_kmeans, calinski_kmeans = evaluate_clustering(X_scaled, kmeans_labels)
print(f"\nK-means Clustering Results:")
print(f"Silhouette Score: {silhouette_kmeans:.4f}")
print(f"Calinski-Harabasz Score: {calinski_kmeans:.4f}")

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Explained Variance Ratio')
plt.savefig('pca_variance_ratio.png')
plt.close()

# Perform K-means on PCA-reduced data (using 2 components)
n_components = 2
pca_reduced = PCA(n_components=n_components)
X_pca_reduced = pca_reduced.fit_transform(X_scaled)

kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_pca_labels = kmeans_pca.fit_predict(X_pca_reduced)

# Evaluate PCA-reduced K-means clustering
silhouette_pca, calinski_pca = evaluate_clustering(X_pca_reduced, kmeans_pca_labels)
print(f"\nPCA-reduced K-means Clustering Results:")
print(f"Silhouette Score: {silhouette_pca:.4f}")
print(f"Calinski-Harabasz Score: {calinski_pca:.4f}")

# Visualize clusters in PCA space
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca_reduced[:, 0], X_pca_reduced[:, 1], 
                     c=kmeans_pca_labels, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Clusters in PCA Space')
plt.colorbar(scatter)
plt.savefig('pca_clusters.png')
plt.close()

# Perform hierarchical clustering
Z = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.savefig('dendrogram.png')
plt.close()

# Create interactive visualizations using plotly
# 3D scatter plot of original features
fig = px.scatter_3d(df, x='exercise_time', y='sleep_hours', z='bmi',
                    color=kmeans_labels, title='3D Visualization of Clusters')
fig.write_html('3d_clusters.html')

# Parallel coordinates plot
df['cluster'] = kmeans_labels
fig = px.parallel_coordinates(df, color='cluster',
                            title='Parallel Coordinates Plot of Health Indicators')
fig.write_html('parallel_coordinates.html') 