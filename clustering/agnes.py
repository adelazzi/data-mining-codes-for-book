# simple_agnes.py - Simplified AGNES with Basic Menu

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

class SimpleAGNES:
    """
    A simplified implementation of AGNES clustering with a basic menu system.
    """
    
    def __init__(self):
        self.X = None
        self.linkage_matrix = None
        self.current_k = 3
        
    def fit(self, X):
        """
        Fit the AGNES algorithm to the data.
        
        Parameters:
        X (array-like): Input data to cluster
        """
        self.X = X
        # Use scipy's linkage function for simplicity and reliability
        distances = pdist(X, metric='euclidean')
        self.linkage_matrix = linkage(distances, method='ward')
        
    def get_clusters(self, k):
        """
        Get cluster labels for k clusters.
        
        Parameters:
        k (int): Number of clusters
        
        Returns:
        array: Cluster labels
        """
        from scipy.cluster.hierarchy import fcluster
        labels = fcluster(self.linkage_matrix, k, criterion='maxclust')
        return labels - 1  # Convert to 0-based indexing
    
    def plot_scatter(self, k):
        """
        Plot scatter plot of clusters.
        
        Parameters:
        k (int): Number of clusters
        """
        labels = self.get_clusters(k)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=labels, cmap='viridis', s=60, alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'AGNES Clustering - {k} Clusters')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_dendrogram(self):
        """
        Plot dendrogram with cut line for current k.
        """
        plt.figure(figsize=(12, 8))
        
        # Plot dendrogram
        dend = dendrogram(self.linkage_matrix, leaf_rotation=90)
        
        # Add cut line for current k
        if self.current_k > 1:
            # Find the height to cut for k clusters
            heights = self.linkage_matrix[:, 2]
            cut_height = heights[-(self.current_k-1)]
            plt.axhline(y=cut_height, color='red', linestyle='--', linewidth=2,
                       label=f'Cut for {self.current_k} clusters')
            plt.legend()
        
        plt.title(f'AGNES Dendrogram (Cut for {self.current_k} clusters)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()
    
    def simple_menu(self):
        """
        Simple interactive menu with two main options.
        """
        print("\n🌳 Simple AGNES Clustering Menu")
        print("=" * 35)
        
        while True:
            print(f"\nCurrent number of clusters (k): {self.current_k}")
            print("\nOptions:")
            print("1. Set number of clusters (k)")
            print("2. Show scatter plot")
            print("3. Show dendrogram")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                try:
                    new_k = int(input(f"Enter number of clusters (1-{len(self.X)}): "))
                    if 1 <= new_k <= len(self.X):
                        self.current_k = new_k
                        print(f"✓ Number of clusters set to {new_k}")
                    else:
                        print(f"❌ Please enter a number between 1 and {len(self.X)}")
                except ValueError:
                    print("❌ Please enter a valid integer")
                    
            elif choice == '2':
                print(f"📊 Showing scatter plot for {self.current_k} clusters...")
                self.plot_scatter(self.current_k)
                
            elif choice == '3':
                print(f"🌳 Showing dendrogram with cut line for {self.current_k} clusters...")
                self.plot_dendrogram()
                
            elif choice == '4':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

def create_sample_data():
    """
    Create sample data for clustering.
    """
    # Create 4 blob clusters for demonstration
    X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.2, 
                      center_box=(-10.0, 10.0), random_state=42)
    return X

if __name__ == "__main__":
    print("🏰 Simple AGNES Hierarchical Clustering")
    print("=" * 40)
    
    # Create sample data
    print("📊 Generating sample data...")
    X = create_sample_data()
    print(f"✓ Created dataset with {len(X)} points")
    
    # Initialize and fit AGNES
    print("🔄 Fitting AGNES model...")
    agnes = SimpleAGNES()
    agnes.fit(X)
    print("✓ Model fitted successfully")
    
    # Show initial scatter plot
    print("📊 Showing initial data...")
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.7, s=50)
    plt.title('Sample Data for AGNES Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Launch simple menu
    agnes.simple_menu()