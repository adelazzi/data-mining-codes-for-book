# naive_bayes.py - The Simple Yet Powerful Prophecy

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class NaiveBayes:
    """
    A from-scratch implementation of Gaussian Naive Bayes classifier.
    This algorithm applies Bayes' theorem with the "naive" assumption of feature independence.
    """
    
    def __init__(self):
        """Initialize the Naive Bayes classifier."""
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
        
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.
        
        Parameters:
        X (array-like): Training features of shape (n_samples, n_features)
        y (array-like): Training labels of shape (n_samples,)
        """
        # Get unique classes
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        # Initialize arrays to store mean and variance for each class and feature
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        # Calculate mean, variance, and prior for each class
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i, :] = X_c.mean(axis=0)
            self.var[i, :] = X_c.var(axis=0)
            self.priors[i] = X_c.shape[0] / X.shape[0]
            
        return self
    
    def _pdf(self, class_idx, x):
        """
        Calculate the probability density function for a feature vector given a class.
        
        Parameters:
        class_idx (int): Index of the class
        x (array-like): Feature vector
        
        Returns:
        float: Probability density
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        
        # Gaussian probability density function
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        
        return numerator / denominator
    
    def _predict_instance(self, x):
        """
        Predict the class for a single instance.
        
        Parameters:
        x (array-like): A single data point
        
        Returns:
        int: Predicted class label
        """
        posteriors = []
        
        # Calculate posterior probability for each class
        for i, c in enumerate(self.classes):
            # Prior probability
            prior = np.log(self.priors[i])
            # Conditional probability (using log to avoid underflow)
            conditional = np.sum(np.log(self._pdf(i, x)))
            # Posterior probability
            posterior = prior + conditional
            posteriors.append(posterior)
            
        # Return the class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        X (array-like): Test data of shape (n_samples, n_features)
        
        Returns:
        array: Predicted class labels
        """
        predictions = [self._predict_instance(x) for x in X]
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
        
        print(f"Naive Bayes Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return accuracy

# Example usage with Iris dataset
if __name__ == "__main__":
    print("🔮 Unleashing the Simple Yet Powerful Prophecy of Naive Bayes...")
    
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize the features (not strictly necessary for Naive Bayes but often helpful)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create and train the Naive Bayes classifier
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = nb.evaluate(X_test, y_test)
    
    # Make a prediction on a new sample
    print("\n🔮 Making a prediction for a new flower...")
    new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # Sepal length, sepal width, petal length, petal width
    new_flower = scaler.transform(new_flower)  # Scale the new sample
    
    prediction = nb.predict(new_flower)
    print(f"New flower measurements: {new_flower}")
    print(f"Predicted class: {prediction[0]} ({iris.target_names[prediction[0]]})")
    
    # Compare with sklearn's implementation for verification
    print("\n" + "="*50)
    print("Comparing with sklearn's Naive Bayes implementation...")
    
    from sklearn.naive_bayes import GaussianNB
    
    sk_nb = GaussianNB()
    sk_nb.fit(X_train, y_train)
    sk_predictions = sk_nb.predict(X_test)
    sk_accuracy = accuracy_score(y_test, sk_predictions)
    
    print(f"Sklearn Naive Bayes Accuracy: {sk_accuracy:.4f}")
    
    # Check if our implementation matches sklearn's
    our_predictions = nb.predict(X_test)
    agreement = np.mean(our_predictions == sk_predictions)
    print(f"Agreement between our implementation and sklearn: {agreement:.4f}")