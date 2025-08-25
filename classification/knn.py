# knn.py - The Wisdom of the Nearest Neighbors

import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class KNearestNeighbors:
    """
    A from-scratch implementation of the K-Nearest Neighbors algorithm.
    This class embodies the wisdom of finding the most similar examples in the training data.
    """
    
    def __init__(self, k=3):
        """
        Initialize the KNN classifier.
        
        Parameters:
        k (int): The number of neighbors to consult for prediction.
        """
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """
        Memorize the training data.
        
        Parameters:
        X (array-like): Training features
        y (array-like): Training labels
        
        Returns:
        self: The fitted KNN instance
        """
        # Store the training data (KNN is a lazy learner - it just remembers everything)
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
        
    def predict(self, X):
        """
        Predict the class labels for the provided data.
        
        Parameters:
        X (array-like): Data to be predicted
        
        Returns:
        array: Predicted class labels
        """
        X = np.array(X)
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        """
        Helper method to predict a single instance.
        """
        # Calculate distances from x to all points in the training set
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label (majority vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def _euclidean_distance(self, x1, x2):
        """
        Calculate the Euclidean distance between two points.
        
        Parameters:
        x1, x2 (array-like): Points to calculate distance between
        
        Returns:
        float: Euclidean distance
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance on test data.
        
        Parameters:
        X_test (array-like): Test features
        y_test (array-like): True test labels
        
        Returns:
        float: Accuracy score
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"KNN (k={self.k}) Accuracy: {accuracy:.4f}")
        return accuracy

def plot_decision_boundaries(X, y, model, title="KNN Decision Boundaries"):
    """
    Visualize the decision boundaries of the KNN classifier.
    
    Parameters:
    X (array-like): Feature data (2D for visualization)
    y (array-like): Target labels
    model: The trained KNN model
    title (str): Plot title
    """
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    # Plot the decision boundary
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Example usage
if __name__ == "__main__":
    print("🧙‍♂️ Consulting the Wisdom of the Nearest Neighbors...")
    
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # For visualization purposes, we'll use only the first two features
    X = X[:, :2]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize the features (important for distance-based algorithms)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train and evaluate KNN with different k values
    k_values = [1, 3, 5, 7, 9]
    accuracies = []
    
    for k in k_values:
        print(f"\n--- Training KNN with k={k} ---")
        knn = KNearestNeighbors(k=k)
        knn.fit(X_train, y_train)
        accuracy = knn.evaluate(X_test, y_test)
        accuracies.append(accuracy)
        
        # Plot decision boundaries for k=3
        if k == 3:
            plot_decision_boundaries(X_train, y_train, knn, 
                                    title=f"KNN (k={k}) Decision Boundaries on Iris Dataset")
    
    # Plot accuracy vs k values
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title('KNN Accuracy vs. k Value')
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()
    
    # Find the optimal k
    optimal_k = k_values[np.argmax(accuracies)]
    print(f"\nOptimal k value: {optimal_k} with accuracy: {max(accuracies):.4f}")
    
    # Demonstrate prediction on a new sample
    print("\n🔮 Making a prediction for a new flower...")
    new_flower = np.array([[5.1, 3.5]])  # Sepal length, sepal width
    new_flower = scaler.transform(new_flower)  # Don't forget to scale!
    
    knn_optimal = KNearestNeighbors(k=optimal_k)
    knn_optimal.fit(X_train, y_train)
    prediction = knn_optimal.predict(new_flower)
    
    print(f"New flower measurements: {new_flower}")
    print(f"Predicted class: {prediction[0]} ({iris.target_names[prediction[0]]})")