# pam.py - The Discerning Medoid Summoner

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class PAM:
    """
    A from-scratch implementation of the Partitioning Around Medoids (PAM) algorithm.
    This algorithm is a robust alternative to K-Means that uses actual data points as cluster centers.
    """
    
    def __init__(self, k=3, max_iterations=100, random_state=None):
        """
        Initialize the PAM clustering algorithm.
        
        Parameters:
        k (int): Number of clusters to form
        max_iterations (int): Maximum number of iterations to run the algorithm
        random_state (int): Seed for random number generation for reproducibility
        """
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.medoids = None
        self.labels = None
        self.cost = None
        
    def _calculate_distance(self, X):
        """
        Calculate the pairwise distance matrix between all points.
        
        Parameters:
        X (array-like): Input data
        
        Returns:
        array: Distance matrix
        """
        n_samples = X.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])
        
        return distance_matrix
    
    def _initialize_medoids(self, X, distance_matrix):
        """
        Initialize medoids using the BUILD phase of PAM.
        
        Parameters:
        X (array-like): Input data
        distance_matrix (array-like): Precomputed distance matrix
        
        Returns:
        array: Initial medoids
        """
        n_samples = X.shape[0]
        
        # Choose the first medoid as the point with minimum total distance to all other points
        total_distances = np.sum(distance_matrix, axis=1)
        first_medoid = np.argmin(total_distances)
        medoids = [first_medoid]
        
        # Select remaining medoids
        for _ in range(1, self.k):
            # Calculate the cost of adding each non-medoid point
            costs = np.zeros(n_samples)
            
            for i in range(n_samples):
                if i not in medoids:
                    # Calculate the cost if we add point i as a medoid
                    cost = 0
                    for j in range(n_samples):
                        if j not in medoids:
                            # Find the current closest medoid
                            current_min_dist = min([distance_matrix[j, m] for m in medoids])
                            # Calculate the new distance if we add point i
                            new_dist = distance_matrix[j, i]
                            # Add the improvement (or deterioration) to the cost
                            cost += max(0, current_min_dist - new_dist)
                    
                    costs[i] = cost
            
            # Select the point that minimizes the cost the most
            new_medoid = np.argmax(costs)  # We want to maximize cost reduction (negative cost)
            medoids.append(new_medoid)
        
        return medoids
    
    def _assign_clusters(self, distance_matrix, medoids):
        """
        Assign each data point to the nearest medoid.
        
        Parameters:
        distance_matrix (array-like): Precomputed distance matrix
        medoids (array-like): Current medoids
        
        Returns:
        tuple: Cluster assignments and total cost
        """
        n_samples = distance_matrix.shape[0]
        assignments = np.zeros(n_samples, dtype=int)
        total_cost = 0
        
        for i in range(n_samples):
            # Calculate distances to all medoids
            distances = [distance_matrix[i, m] for m in medoids]
            # Assign to the closest medoid
            assignments[i] = np.argmin(distances)
            # Add to total cost
            total_cost += min(distances)
        
        return assignments, total_cost
    
    def _swap_medoids(self, distance_matrix, medoids, assignments, current_cost):
        """
        Try to improve the clustering by swapping medoids with non-medoids.
        
        Parameters:
        distance_matrix (array-like): Precomputed distance matrix
        medoids (array-like): Current medoids
        assignments (array-like): Current cluster assignments
        current_cost (float): Current total cost
        
        Returns:
        tuple: Updated medoids, assignments, cost, and whether improvement was made
        """
        n_samples = distance_matrix.shape[0]
        improved = False
        best_medoids = medoids.copy()
        best_assignments = assignments.copy()
        best_cost = current_cost
        
        # Try swapping each medoid with each non-medoid
        for medoid_idx in range(len(medoids)):
            for non_medoid in range(n_samples):
                if non_medoid not in medoids:
                    # Create a new set of medoids by swapping
                    new_medoids = medoids.copy()
                    new_medoids[medoid_idx] = non_medoid
                    
                    # Assign points to the new medoids
                    new_assignments, new_cost = self._assign_clusters(distance_matrix, new_medoids)
                    
                    # Check if this swap improves the cost
                    if new_cost < best_cost:
                        best_medoids = new_medoids
                        best_assignments = new_assignments
                        best_cost = new_cost
                        improved = True
        
        return best_medoids, best_assignments, best_cost, improved
    
    def fit(self, X):
        """
        Perform PAM clustering on the data.
        
        Parameters:
        X (array-like): Input data to cluster
        
        Returns:
        self: The fitted PAM instance
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # Calculate distance matrix
        print("Calculating distance matrix...")
        distance_matrix = self._calculate_distance(X)
        
        # Initialize medoids using BUILD phase
        print("Initializing medoids...")
        self.medoids = self._initialize_medoids(X, distance_matrix)
        
        # Assign points to clusters
        self.labels, self.cost = self._assign_clusters(distance_matrix, self.medoids)
        
        # Iterate until convergence or max iterations
        print("Running PAM algorithm...")
        for iteration in range(self.max_iterations):
            # Try to improve by swapping medoids
            new_medoids, new_labels, new_cost, improved = self._swap_medoids(
                distance_matrix, self.medoids, self.labels, self.cost)
            
            if not improved:
                print(f"Converged after {iteration + 1} iterations")
                break
                
            self.medoids = new_medoids
            self.labels = new_labels
            self.cost = new_cost
        
        # Convert medoid indices to actual points
        self.medoids = X[self.medoids]
        
        return self
    
    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters:
        X (array-like): New data to predict
        
        Returns:
        array: Index of the cluster each sample belongs to
        """
        n_samples = X.shape[0]
        n_medoids = self.medoids.shape[0]
        assignments = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Calculate distances to all medoids
            distances = [np.linalg.norm(X[i] - medoid) for medoid in self.medoids]
            # Assign to the closest medoid
            assignments[i] = np.argmin(distances)
        
        return assignments
    
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

def plot_clusters(X, labels, medoids, title="PAM Clustering"):
    """
    Visualize the clusters and medoids.
    
    Parameters:
    X (array-like): Input data
    labels (array-like): Cluster assignments
    medoids (array-like): Cluster medoids
    title (str): Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Plot data points with cluster colors
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
    
    # Plot medoids
    plt.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='X', s=200, label='Medoids', edgecolors='black', linewidth=2)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar(scatter, label='Cluster')
    plt.show()

# Example usage
if __name__ == "__main__":
    print("🧭 The Discerning Medoid Summoner is seeking true representatives...")
    
    # Create sample data with some outliers
    np.random.seed(42)
    X, y_true = make_blobs(n_samples=200, centers=3, cluster_std=0.80, random_state=0)
    
    # Add some outliers
    outliers = np.array([[10, 10], [-10, -10], [10, -10], [-10, 10]])
    X = np.vstack([X, outliers])
    y_true = np.concatenate([y_true, [3, 3, 3, 3]])  # Outliers belong to a different class
    
    # Apply PAM clustering
    pam = PAM(k=3, max_iterations=20, random_state=42)
    pam.fit(X)
    
    # Evaluate the clustering
    silhouette = pam.evaluate(X)
    print(f"Total cost: {pam.cost:.4f}")
    
    # Plot the clusters
    plot_clusters(X, pam.labels, pam.medoids, "PAM Clustering (k=3)")
    
    # Compare with K-Means to show robustness to outliers
    print("\n" + "="*50)
    print("Comparing with K-Means (showing PAM's robustness to outliers)...")
    
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=pam.labels, cmap='viridis', s=50, alpha=0.8)
    plt.scatter(pam.medoids[:, 0], pam.medoids[:, 1], c='red', marker='X', s=200, label='Medoids', edgecolors='black', linewidth=2)
    plt.title(f"PAM Clustering (Silhouette: {silhouette:.4f})")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.8)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids', edgecolors='black', linewidth=2)
    plt.title(f"K-Means Clustering (Silhouette: {kmeans_silhouette:.4f})")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    # Make predictions on new data
    print("\n🔮 Predicting clusters for new data points...")
    new_points = np.array([[0, 0], [5, 5], [-5, -5]])
    predictions = pam.predict(new_points)
    
    for i, point in enumerate(new_points):
        print(f"Point {point} is assigned to cluster {predictions[i]}")
    
    # Show the medoids
    print("\n🏰 The chosen medoids (true representatives):")
    for i, medoid in enumerate(pam.medoids):
        print(f"Medoid {i}: {medoid}")