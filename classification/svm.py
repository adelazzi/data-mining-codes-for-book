# svm.py - The Art of Finding the Perfect Boundary

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import cvxopt  # For quadratic programming optimization
import warnings
warnings.filterwarnings('ignore')

class SupportVectorMachine:
    """
    A from-scratch implementation of a Support Vector Machine classifier.
    This implementation uses quadratic programming to find the optimal hyperplane.
    """
    
    def __init__(self, kernel='linear', C=1.0, gamma=0.1, degree=3):
        """
        Initialize the SVM classifier.
        
        Parameters:
        kernel (str): Type of kernel function ('linear', 'poly', 'rbf')
        C (float): Regularization parameter
        gamma (float): Kernel coefficient for 'rbf' and 'poly' kernels
        degree (int): Degree of the polynomial kernel function
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = 0
        self.X_train = None
        self.y_train = None
        
    def _kernel_function(self, x1, x2):
        """
        Compute the kernel function value between two vectors.
        
        Parameters:
        x1, x2 (array-like): Input vectors
        
        Returns:
        float: Kernel function value
        """
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(x1, x2) + 1) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("Unknown kernel type. Use 'linear', 'poly', or 'rbf'.")
    
    def _compute_kernel_matrix(self, X):
        """
        Compute the kernel matrix for the given data.
        
        Parameters:
        X (array-like): Input data
        
        Returns:
        array: Kernel matrix
        """
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])
        return K
    
    def fit(self, X, y):
        """
        Train the SVM classifier using quadratic programming.
        
        Parameters:
        X (array-like): Training features
        y (array-like): Training labels (should be -1 or 1)
        """
        # Ensure labels are -1 or 1
        y = np.where(y <= 0, -1, 1)
        
        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = y
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(X)
        
        # Set up quadratic programming problem
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n_samples))
        
        # Constraints: A * x <= b
        if self.C is None or self.C == 0:
            # Hard margin SVM
            G = cvxopt.matrix(-np.eye(n_samples))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            # Soft margin SVM
            G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
            h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        
        A = cvxopt.matrix(y.astype(float), (1, n_samples))
        b = cvxopt.matrix(0.0)
        
        # Solve quadratic programming problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # Extract Lagrange multipliers (alphas)
        alphas = np.ravel(solution['x'])
        
        # Find support vectors (alphas > 0)
        sv = alphas > 1e-5
        sv_indices = np.where(sv)[0]
        self.alphas = alphas[sv]
        self.support_vectors = X[sv]
        self.support_vector_labels = y[sv]
        
        # Calculate bias term (b)
        self.b = 0
        for idx, i in enumerate(sv_indices):
            # i is the index of the support vector in the original training set
            self.b += self.support_vector_labels[idx]
            # K[i, sv_indices] gives kernel values between sample i and all support vectors
            self.b -= np.sum(self.alphas * self.support_vector_labels * K[i, sv_indices])
        self.b /= len(self.alphas)
        
        return self
    
    def _decision_function(self, X):
        """
        Calculate the decision function value for each sample.
        
        Parameters:
        X (array-like): Input data
        
        Returns:
        array: Decision function values
        """
        n_samples = X.shape[0]
        decisions = np.zeros(n_samples)
        
        for i in range(n_samples):
            decision = 0
            for alpha, sv_label, sv in zip(self.alphas, self.support_vector_labels, self.support_vectors):
                decision += alpha * sv_label * self._kernel_function(X[i], sv)
            decisions[i] = decision + self.b
        
        return decisions
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        X (array-like): Test data
        
        Returns:
        array: Predicted class labels (-1 or 1)
        """
        decisions = self._decision_function(X)
        return np.sign(decisions)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance on test data.
        
        Parameters:
        X_test (array-like): Test features
        y_test (array-like): True test labels
        
        Returns:
        float: Accuracy score
        """
        # Ensure labels are -1 or 1
        y_test = np.where(y_test <= 0, -1, 1)
        
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"SVM ({self.kernel} kernel) Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=['Class -1', 'Class 1']))
        
        return accuracy

def plot_decision_boundary(X, y, model, title="SVM Decision Boundary"):
    """
    Visualize the decision boundary of the SVM classifier.
    
    Parameters:
    X (array-like): Feature data (2D for visualization)
    y (array-like): Target labels
    model: The trained SVM model
    title (str): Plot title
    """
    # Create a mesh to plot in
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
    plt.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], 
                s=100, facecolors='none', edgecolors='k', marker='o', linewidths=1.5)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

# Example usage
if __name__ == "__main__":
    print("⚔️  Mastering the Art of Finding the Perfect Boundary with SVM...")
    
    # Create a linearly separable dataset
    print("\n--- Linear SVM on Linearly Separable Data ---")
    X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)
    y = np.where(y == 0, -1, 1)  # Convert to -1, 1 labels
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train and evaluate linear SVM
    linear_svm = SupportVectorMachine(kernel='linear', C=1.0)
    linear_svm.fit(X_train, y_train)
    linear_accuracy = linear_svm.evaluate(X_test, y_test)
    
    # Plot decision boundary
    plot_decision_boundary(X_train, y_train, linear_svm, 
                          title="Linear SVM Decision Boundary")
    
    # Create a non-linearly separable dataset
    print("\n--- Non-Linear SVM on Circular Data ---")
    X_circle, y_circle = make_circles(n_samples=100, factor=0.5, noise=0.1, random_state=42)
    y_circle = np.where(y_circle == 0, -1, 1)  # Convert to -1, 1 labels
    
    # Split the data
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_circle, y_circle, 
                                                               test_size=0.3, random_state=42)
    
    # Standardize the features
    X_train_c = scaler.fit_transform(X_train_c)
    X_test_c = scaler.transform(X_test_c)
    
    # Train and evaluate RBF kernel SVM
    rbf_svm = SupportVectorMachine(kernel='rbf', C=1.0, gamma=0.5)
    rbf_svm.fit(X_train_c, y_train_c)
    rbf_accuracy = rbf_svm.evaluate(X_test_c, y_test_c)
    
    # Plot decision boundary
    plot_decision_boundary(X_train_c, y_train_c, rbf_svm, 
                          title="RBF Kernel SVM Decision Boundary")
    
    # Compare with sklearn's implementation
    print("\n" + "="*50)
    print("Comparing with sklearn's SVM implementation...")
    
    from sklearn.svm import SVC
    
    # Linear kernel
    sk_linear_svm = SVC(kernel='linear', C=1.0)
    sk_linear_svm.fit(X_train, y_train)
    sk_linear_pred = sk_linear_svm.predict(X_test)
    sk_linear_accuracy = accuracy_score(y_test, sk_linear_pred)
    
    print(f"Our Linear SVM Accuracy: {linear_accuracy:.4f}")
    print(f"Sklearn Linear SVM Accuracy: {sk_linear_accuracy:.4f}")
    
    # RBF kernel
    sk_rbf_svm = SVC(kernel='rbf', C=1.0, gamma=0.5)
    sk_rbf_svm.fit(X_train_c, y_train_c)
    sk_rbf_pred = sk_rbf_svm.predict(X_test_c)
    sk_rbf_accuracy = accuracy_score(y_test_c, sk_rbf_pred)
    
    print(f"Our RBF SVM Accuracy: {rbf_accuracy:.4f}")
    print(f"Sklearn RBF SVM Accuracy: {sk_rbf_accuracy:.4f}")
    
    # Make a prediction on a new sample
    print("\n🔮 Making a prediction for a new data point...")
    new_sample = np.array([[1.5, 2.0]])
    new_sample = scaler.transform(new_sample)
    
    prediction = linear_svm.predict(new_sample)
    print(f"New sample: {new_sample}")
    print(f"Predicted class: {prediction[0]}")