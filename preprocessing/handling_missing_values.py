# handling_missing_values.py - The Art of Summoning Lost Data

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def detect_missing_values(df):
    """
    Casts the Void Detection spell.
    Reveals the presence and pattern of missing values in the dataset.

    Parameters:
    df (DataFrame): The dataset to examine.

    Returns:
    Series: A summary of missing values per column.
    """
    print("🧐 Scanning for voids in the data...")
    missing_summary = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    missing_info = pd.DataFrame({
        'Missing Values': missing_summary,
        'Percentage (%)': missing_percentage.round(2)
    })
    
    print(f"Total cells touched by void: {df.isnull().sum().sum()}")
    print(f"Percentage of dataset void: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100).round(2)}%")
    print("\nVoid by column:")
    return missing_info[missing_info['Missing Values'] > 0]

def remove_missing_data(df, axis=0, threshold=0.3):
    """
    Performs the Excision Ritual.
    Removes rows or columns based on their missing value threshold.

    Parameters:
    df (DataFrame): The dataset to purify.
    axis (int): 0 to remove rows, 1 to remove columns.
    threshold (float): Maximum percentage of missing values allowed.

    Returns:
    DataFrame: The purified dataset.
    """
    print("⚔️  Excising tainted elements...")
    if axis == 0:
        # Remove rows with more than threshold% missing values
        thresh = df.shape[1] * (1 - threshold)
        return df.dropna(axis=0, thresh=thresh)
    else:
        # Remove columns with more than threshold% missing values
        thresh = df.shape[0] * (1 - threshold)
        return df.dropna(axis=1, thresh=thresh)

def impute_numerical(df, columns, strategy='mean', fill_value=None):
    """
    Uses the Numerical Imputation glyph.
    Fills missing numerical values with specified strategy.

    Parameters:
    df (DataFrame): The dataset to purify.
    columns (list): Numerical columns to impute.
    strategy (str): 'mean', 'median', 'constant', or 'mode'.
    fill_value: Value to use when strategy='constant'.

    Returns:
    DataFrame: The dataset with imputed numerical values.
    """
    print(f"🔢 Imputing numerical voids with {strategy}...")
    df_imputed = df.copy()
    
    for col in columns:
        if strategy == 'mean':
            fill_val = df[col].mean()
        elif strategy == 'median':
            fill_val = df[col].median()
        elif strategy == 'mode':
            fill_val = df[col].mode()[0]
        elif strategy == 'constant':
            fill_val = fill_value
        else:
            raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'constant'")
        
        df_imputed[col].fillna(fill_val, inplace=True)
    
    return df_imputed

def impute_categorical(df, columns, strategy='most_frequent', fill_value='Missing'):
    """
    Uses the Categorical Imputation glyph.
    Fills missing categorical values with specified strategy.

    Parameters:
    df (DataFrame): The dataset to purify.
    columns (list): Categorical columns to impute.
    strategy (str): 'most_frequent' or 'constant'.
    fill_value: Value to use when strategy='constant'.

    Returns:
    DataFrame: The dataset with imputed categorical values.
    """
    print(f"🔤 Imputing categorical voids with {strategy}...")
    df_imputed = df.copy()
    
    for col in columns:
        if strategy == 'most_frequent':
            fill_val = df[col].mode()[0]
        elif strategy == 'constant':
            fill_val = fill_value
        else:
            raise ValueError("Strategy must be 'most_frequent' or 'constant'")
        
        df_imputed[col].fillna(fill_val, inplace=True)
    
    return df_imputed

def predictive_imputation(df, target_column):
    """
    Uses the Advanced Predictive Imputation spell.
    Predicts missing values using other features as predictors.

    Parameters:
    df (DataFrame): The dataset to purify.
    target_column (str): The column with missing values to impute.

    Returns:
    DataFrame: The dataset with predictively imputed values.
    """
    print(f"🔮 Predictively imputing {target_column} using other features...")
    
    # Separate data into known and unknown values
    known_data = df[df[target_column].notnull()]
    unknown_data = df[df[target_column].isnull()]
    
    # Features to use for prediction (all other numerical columns)
    feature_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                      if col != target_column and not df[col].isnull().any()]
    
    if not feature_columns:
        print("No valid features found for predictive imputation. Falling back to median.")
        return impute_numerical(df, [target_column], strategy='median')
    
    # Prepare training and prediction data
    X_train = known_data[feature_columns]
    y_train = known_data[target_column]
    X_pred = unknown_data[feature_columns]
    
    # Train a model (Random Forest in this case)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict the missing values
    predicted_values = model.predict(X_pred)
    
    # Fill in the missing values
    df_imputed = df.copy()
    df_imputed.loc[df[target_column].isnull(), target_column] = predicted_values
    
    return df_imputed

# Example Ritual Casting (for testing)
if __name__ == "__main__":
    print("=" * 60)
    print("BEGINNING THE RITUAL OF SUMMONING LOST DATA")
    print("=" * 60)
    
    # Create a sample dataset with various types of missing values
    np.random.seed(42)
    sample_data = {
        'age': [25, 32, np.nan, 45, 33, 29, np.nan, 38],
        'income': [50000, 75000, 62000, np.nan, 48000, np.nan, 81000, 67000],
        'department': ['Sales', np.nan, 'IT', 'IT', 'HR', 'HR', np.nan, 'Sales'],
        'performance_score': [4.2, 3.8, np.nan, 4.5, 3.9, 4.1, 4.3, np.nan]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original Dataset (Touched by Void):")
    print(df)
    print("\n")
    
    # 1. Detect missing values
    missing_info = detect_missing_values(df)
    print(missing_info)
    print("\n")
    
    # 2. Remove columns with more than 30% missing values
    df_clean = remove_missing_data(df, axis=1, threshold=0.3)
    print("After removing columns with >30% void:")
    print(df_clean)
    print("\n")
    
    # 3. Impute numerical columns
    numerical_cols = ['age', 'income', 'performance_score']
    df_imputed_num = impute_numerical(df, numerical_cols, strategy='median')
    print("After numerical imputation:")
    print(df_imputed_num)
    print("\n")
    
    # 4. Impute categorical columns
    categorical_cols = ['department']
    df_imputed_cat = impute_categorical(df, categorical_cols, strategy='most_frequent')
    print("After categorical imputation:")
    print(df_imputed_cat)
    print("\n")
    
    # 5. Demonstrate predictive imputation (on the original data)
    print("Demonstrating predictive imputation on original data:")
    df_predictive = predictive_imputation(df, 'income')
    print(df_predictive)
    
    print("\n" + "=" * 60)
    print("RITUAL COMPLETE: DATA PURIFIED AND READY FOR DIVINATION")
    print("=" * 60)