# dbscan.py - The Finder of Dense Constellations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import silhouette_score
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class DBSCAN:
    """
    A from-scratch implementation of the DBSCAN clustering algorithm.
    This algorithm finds clusters based on density and identifies outliers as noise.
    """
    
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialize the DBSCAN clustering algorithm.
        
        Parameters:
        eps (float): The maximum distance between two samples for one to be considered
                     as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be
                           considered as a core point.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.core_points = None
        
    def _euclidean_distance(self, x1, x2):
        """
        Compute the Euclidean distance between two points.
        
        Parameters:
        x1, x2 (array-like): Points to calculate distance between
        
        Returns:
        float: Euclidean distance
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _find_neighbors(self, X, point_idx):
        """
        Find all points within epsilon distance of a given point.
        
        Parameters:
        X (array-like): The data matrix
        point_idx (int): Index of the point to find neighbors for
        
        Returns:
        list: Indices of points within epsilon distance
        """
        neighbors = []
        for i in range(X.shape[0]):
            if self._euclidean_distance(X[point_idx], X[i]) <= self.eps:
                neighbors.append(i)
        return neighbors
    
    def fit(self, X):
        """
        Perform DBSCAN clustering on the data.
        
        Parameters:
        X (array-like): Input data to cluster
        
        Returns:
        self: The fitted DBSCAN instance
        """
        n_samples = X.shape[0]
        
        # Initialize labels: 0 means unvisited, -1 means noise
        labels = np.zeros(n_samples, dtype=int)
        core_points = np.zeros(n_samples, dtype=bool)
        
        cluster_id = 0
        
        for i in range(n_samples):
            if labels[i] != 0:  # Already visited
                continue
                
            # Find all neighbors of point i
            neighbors = self._find_neighbors(X, i)
            
            if len(neighbors) < self.min_samples:
                # Mark as noise (may be reassigned later)
                labels[i] = -1
            else:
                # Start a new cluster
                cluster_id += 1
                labels[i] = cluster_id
                core_points[i] = True
                
                # Expand the cluster using a queue
                queue = deque(neighbors)
                
                while queue:
                    # Get the next point from the queue
                    j = queue.popleft()
                    
                    if labels[j] == -1:
                        # Change noise to border point
                        labels[j] = cluster_id
                    elif labels[j] == 0:
                        # Add to cluster
                        labels[j] = cluster_id
                        
                        # Find neighbors of point j
                        j_neighbors = self._find_neighbors(X, j)
                        
                        if len(j_neighbors) >= self.min_samples:
                            # Point j is a core point
                            core_points[j] = True
                            # Add its neighbors to the queue
                            queue.extend(j_neighbors)
        
        self.labels = labels
        self.core_points = core_points
        
        return self
    
    def predict(self, X_new):
        """
        Predict the cluster labels for new data points.
        
        Parameters:
        X_new (array-like): New data to predict
        
        Returns:
        array: Cluster labels for each new point (-1 for noise)
        """
        n_new = X_new.shape[0]
        n_original = self.core_points.sum()  # Number of core points in the original data
        
        if n_original == 0:
            return np.full(n_new, -1)  # All noise if no core points
        
        # Get the core points from the original data
        core_indices = np.where(self.core_points)[0]
        
        labels_new = np.zeros(n_new, dtype=int)
        
        for i in range(n_new):
            # Find the distance to all core points
            distances = [self._euclidean_distance(X_new[i], self.X_core[j]) 
                        for j in range(n_original)]
            
            min_dist = min(distances)
            min_idx = np.argmin(distances)
            
            if min_dist <= self.eps:
                # Assign to the cluster of the nearest core point
                labels_new[i] = self.labels[core_indices[min_idx]]
            else:
                # Mark as noise
                labels_new[i] = -1
        
        return labels_new
    
    def evaluate(self, X):
        """
        Evaluate the clustering using silhouette score (excluding noise points).
        
        Parameters:
        X (array-like): Input data
        
        Returns:
        float: Silhouette score
        """
        # Exclude noise points for evaluation
        non_noise_mask = self.labels != -1
        
        if np.sum(non_noise_mask) <= 1:
            return -1  # Not enough non-noise points
        
        # Calculate silhouette score only on non-noise points
        score = silhouette_score(X[non_noise_mask], self.labels[non_noise_mask])
        print(f"Silhouette Score (excluding noise): {score:.4f}")
        
        return score

def plot_clusters(X, labels, core_points, title="DBSCAN Clustering"):
    """
    Visualize the DBSCAN clusters, core points, and noise.
    
    Parameters:
    X (array-like): Input data
    labels (array-like): Cluster assignments
    core_points (array-like): Boolean array indicating core points
    title (str): Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Create a color map for clusters
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each cluster
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Noise points (black)
            class_mask = (labels == k)
            plt.scatter(X[class_mask, 0], X[class_mask, 1], c='black', s=20, alpha=0.6, marker='x', label='Noise')
        else:
            # Cluster points
            class_mask = (labels == k)
            core_mask = core_points & class_mask
            non_core_mask = class_mask & ~core_points
            
            # Plot non-core points
            plt.scatter(X[non_core_mask, 0], X[non_core_mask, 1], c=[col], s=30, alpha=0.6, label=f'Cluster {k} (border)')
            
            # Plot core points
            plt.scatter(X[core_mask, 0], X[core_mask, 1], c=[col], s=80, alpha=0.9, marker='o', edgecolors='black', linewidth=1, label=f'Cluster {k} (core)')
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def find_optimal_eps(X, min_samples, k=10):
    """
    Find an optimal epsilon value using the k-distance graph.
    
    Parameters:
    X (array-like): Input data
    min_samples (int): The min_samples parameter for DBSCAN
    k (int): The number of nearest neighbors to consider
    
    Returns:
    float: Suggested epsilon value
    """
    n_samples = X.shape[0]
    k_distances = []
    
    for i in range(n_samples):
        # Calculate distances to all other points
        distances = [np.linalg.norm(X[i] - X[j]) for j in range(n_samples) if i != j]
        # Sort distances and take the k-th smallest
        distances.sort()
        k_distances.append(distances[min(k, len(distances)) - 1])
    
    # Sort the k-distances
    k_distances.sort()
    
    # Plot the k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_samples), k_distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-th nearest neighbor distance')
    plt.title('k-Distance Graph for Epsilon Estimation')
    plt.grid(True)
    
    # The "elbow" point is often a good value for epsilon
    # For simplicity, we'll return the value at the 95th percentile
    suggested_eps = np.percentile(k_distances, 95)
    plt.axhline(y=suggested_eps, color='r', linestyle='--', label=f'Suggested eps: {suggested_eps:.2f}')
    plt.legend()
    plt.show()
    
    return suggested_eps

# Example usage
if __name__ == "__main__":
    print("🌌 The Finder of Dense Constellations is exploring the data universe...")
    
    # Create sample data with non-globular clusters
    np.random.seed(42)
    X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
    X_circles, _ = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=0)
    
    # Find optimal epsilon for the moons dataset
    print("Finding optimal epsilon for moons dataset...")
    eps_moons = find_optimal_eps(X_moons, min_samples=5)
    
    # Apply DBSCAN to the moons dataset
    print(f"\n--- DBSCAN on Moons Dataset (eps={eps_moons:.2f}) ---")
    dbscan_moons = DBSCAN(eps=eps_moons, min_samples=5)
    dbscan_moons.fit(X_moons)
    
    # Evaluate the clustering
    silhouette_moons = dbscan_moons.evaluate(X_moons)
    
    # Plot the clusters
    plot_clusters(X_moons, dbscan_moons.labels, dbscan_moons.core_points, 
                 f"DBSCAN on Moons Dataset (eps={eps_moons:.2f})")
    
    # Find optimal epsilon for the circles dataset
    print("\nFinding optimal epsilon for circles dataset...")
    eps_circles = find_optimal_eps(X_circles, min_samples=5)
    
    # Apply DBSCAN to the circles dataset
    print(f"\n--- DBSCAN on Circles Dataset (eps={eps_circles:.2f}) ---")
    dbscan_circles = DBSCAN(eps=eps_circles, min_samples=5)
    dbscan_circles.fit(X_circles)
    
    # Evaluate the clustering
    silhouette_circles = dbscan_circles.evaluate(X_circles)
    
    # Plot the clusters
    plot_clusters(X_circles, dbscan_circles.labels, dbscan_circles.core_points, 
                 f"DBSCAN on Circles Dataset (eps={eps_circles:.2f})")
    
    # Compare with K-Means to show DBSCAN's advantage with non-globular clusters
    print("\n" + "="*50)
    print("Comparing DBSCAN with K-Means on non-globular data...")
    
    from sklearn.cluster import KMeans
    
    kmeans_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_moons_labels = kmeans_moons.fit_predict(X_moons)
    kmeans_moons_silhouette = silhouette_score(X_moons, kmeans_moons_labels)
    
    kmeans_circles = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_circles_labels = kmeans_circles.fit_predict(X_circles)
    kmeans_circles_silhouette = silhouette_score(X_circles, kmeans_circles_labels)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=dbscan_moons.labels, cmap='viridis', s=30)
    plt.title(f"DBSCAN on Moons (Silhouette: {silhouette_moons:.4f})")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.subplot(2, 2, 2)
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=kmeans_moons_labels, cmap='viridis', s=30)
    plt.title(f"K-Means on Moons (Silhouette: {kmeans_moons_silhouette:.4f})")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.subplot(2, 2, 3)
    plt.scatter(X_circles[:, 0], X_circles[:, 1], c=dbscan_circles.labels, cmap='viridis', s=30)
    plt.title(f"DBSCAN on Circles (Silhouette: {silhouette_circles:.4f})")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.subplot(2, 2, 4)
    plt.scatter(X_circles[:, 0], X_circles[:, 1], c=kmeans_circles_labels, cmap='viridis', s=30)
    plt.title(f"K-Means on Circles (Silhouette: {kmeans_circles_silhouette:.4f})")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate noise detection
    print("\n📊 Analyzing the noise points...")
    n_noise_moons = np.sum(dbscan_moons.labels == -1)
    n_noise_circles = np.sum(dbscan_circles.labels == -1)
    
    print(f"Moons dataset: {n_noise_moons} noise points ({n_noise_moons/len(X_moons)*100:.1f}%)")
    print(f"Circles dataset: {n_noise_circles} noise points ({n_noise_circles/len(X_circles)*100:.1f}%)")
    
    # Make predictions on new data
    print("\n🔮 Predicting clusters for new data points...")
    # Store the original data for prediction
    dbscan_moons.X_core = X_moons[dbscan_moons.core_points]
    dbscan_circles.X_core = X_circles[dbscan_circles.core_points]
    
    new_points = np.array([[0, 0], [1.5, 0.2], [-1, 1.5], [0.5, -1]])
    
    moons_predictions = dbscan_moons.predict(new_points)
    circles_predictions = dbscan_circles.predict(new_points)
    
    for i, point in enumerate(new_points):
        print(f"Point {point}:")
        print(f"  Moons prediction: {moons_predictions[i]} ({'Noise' if moons_predictions[i] == -1 else 'Cluster'})")
        print(f"  Circles prediction: {circles_predictions[i]} ({'Noise' if circles_predictions[i] == -1 else 'Cluster'})")
        print()