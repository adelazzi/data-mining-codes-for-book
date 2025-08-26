# clara.py - The Sharp-Eyed Clarity Bringer for Large Realms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class CLARA:
    """
    A from-scratch implementation of the CLARA (Clustering LARge Applications) algorithm.
    This algorithm efficiently clusters large datasets by applying PAM to multiple samples.
    """
    
    def __init__(self, k=3, n_samples=5, sample_size=40, max_iterations=10, random_state=None):
        """
        Initialize the CLARA clustering algorithm.
        
        Parameters:
        k (int): Number of clusters to form
        n_samples (int): Number of samples to draw
        sample_size (int): Size of each sample
        max_iterations (int): Maximum number of iterations for PAM on each sample
        random_state (int): Seed for random number generation for reproducibility
        """
        self.k = k
        self.n_samples = n_samples
        self.sample_size = sample_size
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.medoids = None
        self.labels = None
        self.best_cost = float('inf')
        
    def _calculate_distance(self, x1, x2):
        """
        Compute the Euclidean distance between two points.
        
        Parameters:
        x1, x2 (array-like): Points to calculate distance between
        
        Returns:
        float: Euclidean distance
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _initialize_medoids(self, X, distance_matrix):
        """
        Initialize medoids for a sample using the BUILD phase of PAM.
        
        Parameters:
        X (array-like): Sample data
        distance_matrix (array-like): Precomputed distance matrix for the sample
        
        Returns:
        array: Initial medoids (indices within the sample)
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
    
    def _assign_clusters(self, X, medoids):
        """
        Assign each data point to the nearest medoid.
        
        Parameters:
        X (array-like): Input data
        medoids (array-like): Current medoids (actual points, not indices)
        
        Returns:
        tuple: Cluster assignments and total cost
        """
        n_samples = X.shape[0]
        assignments = np.zeros(n_samples, dtype=int)
        total_cost = 0
        
        for i in range(n_samples):
            # Calculate distances to all medoids
            distances = [self._calculate_distance(X[i], medoid) for medoid in medoids]
            # Assign to the closest medoid
            assignments[i] = np.argmin(distances)
            # Add to total cost
            total_cost += min(distances)
        
        return assignments, total_cost
    
    def _pam_on_sample(self, sample):
        """
        Apply PAM to a sample of the data.
        
        Parameters:
        sample (array-like): A sample from the full dataset
        
        Returns:
        tuple: Medoids from the sample and the cost on the full dataset
        """
        n_sample = sample.shape[0]
        
        # Calculate distance matrix for the sample
        distance_matrix = np.zeros((n_sample, n_sample))
        for i in range(n_sample):
            for j in range(n_sample):
                distance_matrix[i, j] = self._calculate_distance(sample[i], sample[j])
        
        # Initialize medoids
        medoid_indices = self._initialize_medoids(sample, distance_matrix)
        medoids = sample[medoid_indices]
        
        # Apply PAM iterations
        for iteration in range(self.max_iterations):
            # Assign points in the sample to clusters
            sample_assignments, sample_cost = self._assign_clusters(sample, medoids)
            
            # Try to improve by swapping medoids
            improved = False
            for medoid_idx in range(self.k):
                for non_medoid in range(n_sample):
                    if non_medoid not in medoid_indices:
                        # Create a new set of medoids by swapping
                        new_medoids = medoids.copy()
                        new_medoids[medoid_idx] = sample[non_medoid]
                        
                        # Assign points to the new medoids
                        new_assignments, new_cost = self._assign_clusters(sample, new_medoids)
                        
                        # Check if this swap improves the cost
                        if new_cost < sample_cost:
                            medoids = new_medoids
                            medoid_indices[medoid_idx] = non_medoid
                            improved = True
            
            if not improved:
                break
        
        return medoids
    
    def fit(self, X):
        """
        Perform CLARA clustering on the data.
        
        Parameters:
        X (array-like): Input data to cluster
        
        Returns:
        self: The fitted CLARA instance
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # Ensure sample size is reasonable
        if self.sample_size > n_samples:
            self.sample_size = n_samples
        
        # Draw multiple samples and apply PAM to each
        best_medoids = None
        best_cost = float('inf')
        best_assignments = None
        
        for sample_idx in range(self.n_samples):
            print(f"Processing sample {sample_idx + 1}/{self.n_samples}...")
            
            # Draw a random sample
            sample_indices = np.random.choice(n_samples, self.sample_size, replace=False)
            sample = X[sample_indices]
            
            # Apply PAM to the sample
            sample_medoids = self._pam_on_sample(sample)
            
            # Evaluate these medoids on the full dataset
            assignments, cost = self._assign_clusters(X, sample_medoids)
            
            # Keep track of the best clustering
            if cost < best_cost:
                best_cost = cost
                best_medoids = sample_medoids
                best_assignments = assignments
        
        # Store the best results
        self.medoids = best_medoids
        self.labels = best_assignments
        self.best_cost = best_cost
        
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
        assignments = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Calculate distances to all medoids
            distances = [self._calculate_distance(X[i], medoid) for medoid in self.medoids]
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
        print(f"Total cost: {self.best_cost:.4f}")
        return score

def plot_clusters(X, labels, medoids, title="CLARA Clustering"):
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
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)
    
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
    print("🔍 The Sharp-Eyed Clarity Bringer is surveying the large realm...")
    
    # Create a larger dataset
    np.random.seed(42)
    X, y_true = make_blobs(n_samples=1000, centers=4, cluster_std=0.80, random_state=0)
    
    # Apply CLARA clustering
    clara = CLARA(k=4, n_samples=5, sample_size=100, max_iterations=10, random_state=42)
    clara.fit(X)
    
    # Evaluate the clustering
    silhouette = clara.evaluate(X)
    
    # Plot the clusters
    plot_clusters(X, clara.labels, clara.medoids, "CLARA Clustering (k=4)")
    
    # Compare with PAM on a small subset to show computational efficiency
    print("\n" + "="*50)
    print("Comparing CLARA with PAM on a small subset...")
    
    # Take a small subset for PAM comparison
    subset_indices = np.random.choice(X.shape[0], 200, replace=False)
    X_subset = X[subset_indices]
    
    # Time both algorithms
    import time
    
    # CLARA on the full dataset
    start_time = time.time()
    clara_full = CLARA(k=4, n_samples=5, sample_size=100, max_iterations=10, random_state=42)
    clara_full.fit(X)
    clara_time = time.time() - start_time
    
    # PAM on the subset
    start_time = time.time()
    from pam import PAM  # Assuming we have the PAM implementation from earlier
    pam_subset = PAM(k=4, max_iterations=10, random_state=42)
    pam_subset.fit(X_subset)
    pam_time = time.time() - start_time
    
    print(f"CLARA on full dataset (n=1000): {clara_time:.4f} seconds")
    print(f"PAM on subset (n=200): {pam_time:.4f} seconds")
    print(f"CLARA is {pam_time/clara_time:.2f}x faster on this example")
    
    # Compare quality
    clara_silhouette = silhouette_score(X, clara.labels)
    pam_silhouette = silhouette_score(X_subset, pam_subset.labels)
    
    print(f"CLARA Silhouette Score: {clara_silhouette:.4f}")
    print(f"PAM Silhouette Score: {pam_silhouette:.4f}")
    
    # Show the samples used by CLARA
    print("\n📊 Showing the sampling strategy of CLARA...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c='gray', s=10, alpha=0.3, label='Full dataset')
    
    # Draw one sample
    sample_indices = np.random.choice(X.shape[0], 100, replace=False)
    plt.scatter(X[sample_indices, 0], X[sample_indices, 1], c='blue', s=30, alpha=0.7, label='Sample')
    
    plt.title("CLARA Sampling Strategy")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=clara.labels, cmap='viridis', s=10, alpha=0.6)
    plt.scatter(clara.medoids[:, 0], clara.medoids[:, 1], c='red', marker='X', s=200, label='Medoids', edgecolors='black', linewidth=2)
    plt.title("CLARA Final Clustering")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Make predictions on new data
    print("\n🔮 Predicting clusters for new data points...")
    new_points = np.array([[0, 0], [5, 5], [-5, -5], [10, 10]])
    predictions = clara.predict(new_points)
    
    for i, point in enumerate(new_points):
        print(f"Point {point} is assigned to cluster {predictions[i]}")
    
    # Show the medoids
    print("\n🏰 The chosen medoids (representative points):")
    for i, medoid in enumerate(clara.medoids):
        print(f"Medoid {i}: {medoid}")