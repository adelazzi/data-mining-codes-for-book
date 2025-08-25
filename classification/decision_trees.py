# decision_trees.py - The Branching Path of Choices

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
from collections import Counter

class Node:
    """
    A class representing a node in the decision tree.
    Each node can be either a decision node or a leaf node.
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # For decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # For leaf node
        self.value = value

class DecisionTree:
    """
    A from-scratch implementation of a Decision Tree classifier.
    This implementation uses the CART (Classification and Regression Trees) algorithm.
    """
    
    def __init__(self, min_samples_split=2, max_depth=2, criterion='gini'):
        """
        Initialize the Decision Tree classifier.
        
        Parameters:
        min_samples_split (int): Minimum number of samples required to split a node
        max_depth (int): Maximum depth of the tree
        criterion (str): Splitting criterion ('gini' or 'entropy')
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None
        
    def _gini_index(self, y):
        """
        Calculate the Gini impurity of a set of labels.
        
        Parameters:
        y (array-like): Target labels
        
        Returns:
        float: Gini impurity
        """
        class_counts = np.bincount(y)
        class_probabilities = class_counts / len(y)
        return 1 - np.sum(class_probabilities ** 2)
    
    def _entropy(self, y):
        """
        Calculate the entropy of a set of labels.
        
        Parameters:
        y (array-like): Target labels
        
        Returns:
        float: Entropy
        """
        class_counts = np.bincount(y)
        class_probabilities = class_counts / len(y)
        class_probabilities = class_probabilities[class_probabilities > 0]  # Avoid log(0)
        return -np.sum(class_probabilities * np.log2(class_probabilities))
    
    def _information_gain(self, parent, left_child, right_child, mode='gini'):
        """
        Calculate the information gain from a split.
        
        Parameters:
        parent (array-like): Labels of the parent node
        left_child (array-like): Labels of the left child node
        right_child (array-like): Labels of the right child node
        mode (str): 'gini' for Gini impurity, 'entropy' for information gain
        
        Returns:
        float: Information gain
        """
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        
        if mode == 'gini':
            return self._gini_index(parent) - (weight_left * self._gini_index(left_child) + weight_right * self._gini_index(right_child))
        else:
            return self._entropy(parent) - (weight_left * self._entropy(left_child) + weight_right * self._entropy(right_child))
    
    def _best_split(self, X, y):
        """
        Find the best split for a node.
        
        Parameters:
        X (array-like): Feature matrix
        y (array-like): Target labels
        
        Returns:
        dict: Information about the best split
        """
        best_split = {}
        max_info_gain = -float('inf')
        n_samples, n_features = X.shape
        
        # Check if we have enough samples to split
        if n_samples < self.min_samples_split:
            return best_split
        
        # Calculate the current impurity
        if self.criterion == 'gini':
            current_impurity = self._gini_index(y)
        else:
            current_impurity = self._entropy(y)
        
        # Loop through all features
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)
            
            # Loop through all unique values of the feature
            for threshold in unique_values:
                # Split the dataset based on the threshold
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                # Calculate information gain
                info_gain = self._information_gain(y, y[left_indices], y[right_indices], self.criterion)
                
                # Update best split if needed
                if info_gain > max_info_gain:
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices,
                        'info_gain': info_gain
                    }
                    max_info_gain = info_gain
        
        return best_split
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        
        Parameters:
        X (array-like): Feature matrix
        y (array-like): Target labels
        depth (int): Current depth of the tree
        
        Returns:
        Node: The root node of the tree
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        if (depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)
        
        # Find the best split
        best_split = self._best_split(X, y)
        
        # Check if we found a valid split
        if not best_split or best_split['info_gain'] == 0:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)
        
        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(X[best_split['left_indices']], y[best_split['left_indices']], depth + 1)
        right_subtree = self._build_tree(X[best_split['right_indices']], y[best_split['right_indices']], depth + 1)
        
        # Return the decision node
        return Node(
            feature_index=best_split['feature_index'],
            threshold=best_split['threshold'],
            left=left_subtree,
            right=right_subtree,
            info_gain=best_split['info_gain']
        )
    
    def _calculate_leaf_value(self, y):
        """
        Calculate the value (prediction) for a leaf node.
        
        Parameters:
        y (array-like): Target labels
        
        Returns:
        int: The most common class label
        """
        return Counter(y).most_common(1)[0][0]
    
    def fit(self, X, y):
        """
        Train the decision tree classifier.
        
        Parameters:
        X (array-like): Training features
        y (array-like): Training labels
        """
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x, node):
        """
        Predict the class for a single instance.
        
        Parameters:
        x (array-like): A single data point
        node (Node): Current node in the tree
        
        Returns:
        int: Predicted class label
        """
        if node.value is not None:
            return node.value
        
        feature_value = x[node.feature_index]
        
        if feature_value <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        X (array-like): Test data
        
        Returns:
        array: Predicted class labels
        """
        predictions = [self._predict_single(x, self.root) for x in X]
        return np.array(predictions)
    
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
        
        print(f"Decision Tree Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        return accuracy
    
    def print_tree(self, node=None, depth=0, feature_names=None):
        """
        Print the decision tree structure.
        
        Parameters:
        node (Node): Current node (default is root)
        depth (int): Current depth of the tree
        feature_names (list): Names of the features
        """
        if node is None:
            node = self.root
        
        if node.value is not None:
            print(f"{'  ' * depth}Leaf: Class {node.value}")
        else:
            feature_name = f"Feature {node.feature_index}"
            if feature_names is not None:
                feature_name = feature_names[node.feature_index]
                
            print(f"{'  ' * depth}[{feature_name} <= {node.threshold:.2f}] (IG: {node.info_gain:.4f})")
            self.print_tree(node.left, depth + 1, feature_names)
            self.print_tree(node.right, depth + 1, feature_names)

# Example usage
if __name__ == "__main__":
    print("🌳 Navigating the Branching Path of Choices with Decision Trees...")
    
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and train the Decision Tree classifier
    dt = DecisionTree(min_samples_split=3, max_depth=3, criterion='gini')
    dt.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = dt.evaluate(X_test, y_test)
    
    # Print the tree structure
    print("\n🌳 Decision Tree Structure:")
    dt.print_tree(feature_names=feature_names)
    
    # Compare with sklearn's implementation
    print("\n" + "="*50)
    print("Comparing with sklearn's Decision Tree implementation...")
    
    sk_dt = DecisionTreeClassifier(min_samples_split=3, max_depth=3, random_state=42)
    sk_dt.fit(X_train, y_train)
    sk_predictions = sk_dt.predict(X_test)
    sk_accuracy = accuracy_score(y_test, sk_predictions)
    
    print(f"Our Decision Tree Accuracy: {accuracy:.4f}")
    print(f"Sklearn Decision Tree Accuracy: {sk_accuracy:.4f}")
    
    # Check if our implementation matches sklearn's
    our_predictions = dt.predict(X_test)
    agreement = np.mean(our_predictions == sk_predictions)
    print(f"Agreement between our implementation and sklearn: {agreement:.4f}")
    
    # Visualize the tree using sklearn's plot_tree
    plt.figure(figsize=(12, 8))
    plot_tree(sk_dt, feature_names=feature_names, class_names=target_names, filled=True)
    plt.title("Sklearn Decision Tree Visualization")
    plt.show()
    
    # Make a prediction on a new sample
    print("\n🔮 Making a prediction for a new flower...")
    new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # Sepal length, sepal width, petal length, petal width
    
    prediction = dt.predict(new_flower)
    print(f"New flower measurements: {new_flower}")
    print(f"Predicted class: {prediction[0]} ({target_names[prediction[0]]})")
    
    # Explain the prediction path
    print("\n🧭 Prediction Path:")
    print(f"The decision tree would ask a series of questions about this flower:")
    print(f"1. Is {feature_names[2]} <= {1.9:.2f}? (Petal width)")
    print(f"2. If yes, it would classify as {target_names[0]} (setosa)")
    print(f"3. If no, it would ask additional questions about other features...")