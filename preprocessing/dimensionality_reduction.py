# Z-Score Normalization

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def standardize_data(data):
    """
    Casts the Standardization spell (Z-Score Normalization).
    Centers the data to have a mean of 0 and a standard deviation of 1.

    Parameters:
    data (array-like): The raw, unscaled numerical feature.

    Returns:
    array: The purified, standardized feature.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def normalize_data(data):
    """
    Casts the Min-Max Normalization spell.
    Scales the data to be trapped within the bounds of [0, 1].

    Parameters:
    data (array-like): The raw, unscaled numerical feature.

    Returns:
    array: The feature, confined to a 0-1 range.
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def robust_scale_data(data):
    """
    Casts the Robust Scaling spell.
    Uses the IQR to scale data, making it resilient to outliers.

    Parameters:
    data (array-like): The raw, unscaled numerical feature, possibly with outliers.

    Returns:
    array: The feature, scaled robustly.
    """
    scaler = RobustScaler()
    return scaler.fit_transform(data)

# Example Ritual Casting (for testing)
if __name__ == "__main__":
    # Sample data: a feature where income shouts and age whispers
    raw_data = np.array([[50000, 25],
                         [80000, 40],
                         [120000, 36]]).astype(float)

    print("Original Data (The Unruly Shouts):")
    print(raw_data)

    print("\nAfter Standardization (The Neutral Ground):")
    print(standardize_data(raw_data))

    print("\nAfter Min-Max Normalization (The 0-1 Prison):")
    print(normalize_data(raw_data))