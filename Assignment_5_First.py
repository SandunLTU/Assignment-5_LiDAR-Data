import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator  # For automatic elbow detection

# Load point cloud dataset
file_path = "C:/Users/DELL/Documents/MSc in Maintenance Engineering @LTU/D7015B - Industrial AI and eMaintenance/Assignment 5/Second round in VS/dataset2.npy"
pcd = np.load(file_path)

def get_ground_level(pcd):
    """Finds the best ground level using a histogram."""
    z_values = pcd[:, 2]
    hist, bin_edges = np.histogram(z_values, bins=100)
    return bin_edges[np.argmax(hist)]

def plot_histogram(z_values, ground_level):
    """Plots the histogram of Z-values with the estimated ground level."""
    plt.figure(figsize=(8, 6))
    plt.hist(z_values, bins=100, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(ground_level, color='red', linestyle='dashed', linewidth=2, label=f'Ground Level: {ground_level:.2f}')
    plt.xlabel('Z-value (Height)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Z-values')
    plt.legend()
    plt.grid(True)
    plt.show()

def find_optimal_eps(points):
    """Finds the optimal eps using the k-distance plot and knee detection."""
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors.fit(points)
    distances, _ = neighbors.kneighbors(points)
    distances = np.sort(distances[:, -1])
    
    # Find the knee/elbow point
    kneedle = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
    optimal_eps = distances[kneedle.elbow]  # Get optimal epsilon from the elbow
    
    # Plot elbow method
    plt.figure(figsize=(8, 6))
    plt.plot(distances)
    plt.axvline(kneedle.elbow, color='red', linestyle='dashed', label=f'Optimal eps: {optimal_eps:.2f}')
    plt.xlabel("Points sorted by distance")
    plt.ylabel("5th Nearest Neighbor Distance")
    plt.title("Elbow Method for Optimal eps")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return optimal_eps

# Compute ground level
ground_level = get_ground_level(pcd)
print(f"Estimated Ground Level: {ground_level}")
plot_histogram(pcd[:, 2], ground_level)

# Remove ground points
pcd_above_ground = pcd[pcd[:, 2] > ground_level]

# Find optimal eps
optimal_eps = find_optimal_eps(pcd_above_ground)
print(f"Optimal eps: {optimal_eps}")

# Apply DBSCAN clustering
clustering = DBSCAN(eps=optimal_eps, min_samples=5).fit(pcd_above_ground)
labels = clustering.labels_

# Find the largest cluster (catenary)
unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
largest_cluster_label = unique_labels[np.argmax(counts)]
largest_cluster_points = pcd_above_ground[labels == largest_cluster_label]

# Compute min/max x and y values
min_x, max_x = np.min(largest_cluster_points[:, 0]), np.max(largest_cluster_points[:, 0])
min_y, max_y = np.min(largest_cluster_points[:, 1]), np.max(largest_cluster_points[:, 1])
print(f"Largest Cluster (Catenary) Bounds:")
print(f"Min X: {min_x}, Max X: {max_x}")
print(f"Min Y: {min_y}, Max Y: {max_y}")

# Plot clusters
plt.figure(figsize=(10, 8))
plt.scatter(pcd_above_ground[:, 0], pcd_above_ground[:, 1], c=labels, cmap='viridis', s=2)
plt.title(f'DBSCAN Clustering (eps={optimal_eps:.2f})')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(label='Cluster Label')
plt.show()

# Plot the largest cluster
plt.figure(figsize=(10, 8))
plt.scatter(largest_cluster_points[:, 0], largest_cluster_points[:, 1], color='red', s=2)
plt.title('Largest Cluster (Catenary)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Print number of clusters
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Number of clusters found: {num_clusters}")