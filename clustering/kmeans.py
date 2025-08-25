# kmeans.py - The Seeker of Central Points

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class KMeans:
    """
    A from-scratch implementation of the K-Means clustering algorithm.
    This algorithm partitions data into K clusters by minimizing within-cluster variance.
    """
    
    def __init__(self, k=3, max_iterations=100, tolerance=0.0001):
        """
        Initialize the K-Means clustering algorithm.
        
        Parameters:
        k (int): Number of clusters to form
        max_iterations (int): Maximum number of iterations to run the algorithm
        tolerance (float): Tolerance to declare convergence (if centroids move less than this)
        """
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        
    def _initialize_centroids(self, X):
        """
        Initialize centroids using the k-means++ method for better convergence.
        
        Parameters:
        X (array-like): Input data
        
        Returns:
        array: Initial centroids
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))
        
        # First centroid: choose randomly
        random_idx = np.random.randint(n_samples)
        centroids[0] = X[random_idx]
        
        # Calculate distances to the first centroid
        distances = np.zeros(n_samples)
        
        for i in range(1, self.k):
            # Calculate distances to the closest centroid for each point
            for j in range(n_samples):
                distances[j] = np.min([np.linalg.norm(X[j] - centroids[c])**2 for c in range(i)])
            
            # Choose next centroid with probability proportional to distance squared
            probabilities = distances / np.sum(distances)
            cumulative_prob = np.cumsum(probabilities)
            r = np.random.rand()
            
            for j in range(n_samples):
                if r <= cumulative_prob[j]:
                    centroids[i] = X[j]
                    break
        
        return centroids
    
    def _assign_clusters(self, X, centroids):
        """
        Assign each data point to the nearest centroid.
        
        Parameters:
        X (array-like): Input data
        centroids (array-like): Current centroids
        
        Returns:
        array: Cluster assignments for each data point
        """
        n_samples = X.shape[0]
        assignments = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Calculate distances to all centroids
            distances = [np.linalg.norm(X[i] - centroid) for centroid in centroids]
            # Assign to the closest centroid
            assignments[i] = np.argmin(distances)
        
        return assignments
    
    def _update_centroids(self, X, assignments):
        """
        Update centroids as the mean of assigned data points.
        
        Parameters:
        X (array-like): Input data
        assignments (array-like): Cluster assignments for each data point
        
        Returns:
        array: Updated centroids
        """
        n_features = X.shape[1]
        new_centroids = np.zeros((self.k, n_features))
        
        for i in range(self.k):
            # Get all points assigned to cluster i
            cluster_points = X[assignments == i]
            
            # If cluster has points, calculate mean; otherwise keep old centroid
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # If no points in cluster, reinitialize randomly
                new_centroids[i] = X[np.random.randint(X.shape[0])]
        
        return new_centroids
    
    def _calculate_inertia(self, X, assignments, centroids):
        """
        Calculate the within-cluster sum of squares (inertia).
        
        Parameters:
        X (array-like): Input data
        assignments (array-like): Cluster assignments for each data point
        centroids (array-like): Current centroids
        
        Returns:
        float: Within-cluster sum of squares
        """
        inertia = 0
        for i in range(self.k):
            # Get all points assigned to cluster i
            cluster_points = X[assignments == i]
            # Calculate sum of squared distances to centroid
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[i])**2)
        
        return inertia
    
    def fit(self, X):
        """
        Perform K-Means clustering on the data.
        
        Parameters:
        X (array-like): Input data to cluster
        
        Returns:
        self: The fitted KMeans instance
        """
        n_samples, n_features = X.shape
        
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        # Initialize variables for iteration
        previous_centroids = np.copy(self.centroids)
        self.labels = np.zeros(n_samples, dtype=int)
        
        # Iterate until convergence or max iterations
        for iteration in range(self.max_iterations):
            # Assign points to nearest centroid
            self.labels = self._assign_clusters(X, self.centroids)
            
            # Update centroids
            self.centroids = self._update_centroids(X, self.labels)
            
            # Check for convergence
            centroid_shift = np.sum([np.linalg.norm(self.centroids[i] - previous_centroids[i]) 
                                    for i in range(self.k)])
            
            if centroid_shift < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            previous_centroids = np.copy(self.centroids)
        
        # Calculate final inertia
        self.inertia_ = self._calculate_inertia(X, self.labels, self.centroids)
        
        return self
    
    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters:
        X (array-like): New data to predict
        
        Returns:
        array: Index of the cluster each sample belongs to
        """
        return self._assign_clusters(X, self.centroids)
    
    def evaluate(self, X):
        """
        Evaluate the clustering using silhouette score.
        
        Parameters:
        X (array-like): Input data
        
        Returns:
        float: Silhouette score
        """
        if self.k == 1:
            return 0  # Silhouette score is not defined for k=1
        
        score = silhouette_score(X, self.labels)
        print(f"Silhouette Score: {score:.4f}")
        return score

def plot_clusters(X, labels, centroids, title="K-Means Clustering"):
    """
    Visualize the clusters and centroids.
    
    Parameters:
    X (array-like): Input data
    labels (array-like): Cluster assignments
    centroids (array-like): Cluster centroids
    title (str): Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Plot data points with cluster colors
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar(scatter, label='Cluster')
    plt.show()

def elbow_method(X, k_range=range(1, 11)):
    """
    Use the elbow method to determine the optimal number of clusters.
    
    Parameters:
    X (array-like): Input data
    k_range (range): Range of k values to test
    
    Returns:
    list: Inertia values for each k
    """
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(k=k)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, marker='o', linestyle='-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-cluster sum of squares (Inertia)')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()
    
    return inertias

# Example usage
if __name__ == "__main__":
    print("🧭 The Seeker of Central Points is searching for patterns...")
    
    # Create sample data with clusters
    np.random.seed(42)
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    
    # Apply K-Means clustering
    kmeans = KMeans(k=4)
    kmeans.fit(X)
    
    # Evaluate the clustering
    silhouette = kmeans.evaluate(X)
    
    # Plot the clusters
    plot_clusters(X, kmeans.labels, kmeans.centroids, "K-Means Clustering (k=4)")
    
    # Use the elbow method to find optimal k
    print("\n📈 Using the elbow method to find optimal k...")
    inertias = elbow_method(X)
    
    # Compare with sklearn's implementation
    print("\n" + "="*50)
    print("Comparing with sklearn's K-Means implementation...")
    
    from sklearn.cluster import KMeans as SKKMeans
    
    sk_kmeans = SKKMeans(n_clusters=4, random_state=42, n_init=10)
    sk_labels = sk_kmeans.fit_predict(X)
    sk_inertia = sk_kmeans.inertia_
    sk_silhouette = silhouette_score(X, sk_labels)
    
    print(f"Our K-Means Inertia: {kmeans.inertia_:.4f}")
    print(f"Sklearn K-Means Inertia: {sk_inertia:.4f}")
    print(f"Our Silhouette Score: {silhouette:.4f}")
    print(f"Sklearn Silhouette Score: {sk_silhouette:.4f}")
    
    # Make predictions on new data
    print("\n🔮 Predicting clusters for new data points...")
    new_points = np.array([[0, 2], [8, 3], [-2, -2]])
    predictions = kmeans.predict(new_points)
    
    for i, point in enumerate(new_points):
        print(f"Point {point} is assigned to cluster {predictions[i]}")
    
    # Visualize with new points
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis', s=50, alpha=0.8)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.scatter(new_points[:, 0], new_points[:, 1], c=predictions, cmap='viridis', s=150, marker='*', edgecolors='black', linewidth=2)
    plt.title("K-Means Clustering with New Points")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()