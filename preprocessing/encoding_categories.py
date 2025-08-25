# encoding_categories.py - Translating Ancient Symbols

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

def label_encode(data):
    """
    Uses the Label Encoding glyph.
    Translates categories into simple integers. Use with caution.

    Parameters:
    data (array-like): The categorical feature to be translated.

    Returns:
    array: The integer-encoded feature.
    """
    le = LabelEncoder()
    return le.fit_transform(data)

def one_hot_encode(data):
    """
    Uses the One-Hot Encoding glyph.
    Creates a new binary column for each category. The original feature is consumed in the process.

    Parameters:
    data (array-like): The categorical feature to be translated.

    Returns:
    DataFrame: A new DataFrame with the original feature replaced by binary columns.
    """
    # Ensure input is 2D: convert Series or array-like to a single-column DataFrame
    if isinstance(data, pd.Series):
        data_df = data.to_frame()
    elif isinstance(data, pd.DataFrame):
        data_df = data
    else:
        data_df = pd.DataFrame(data)

    # Initialize the encoder, specifying to handle unknown categories by ignoring them
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = ohe.fit_transform(data_df)  # This returns a NumPy array

    # Create descriptive column names from the DataFrame's column names
    feature_names = ohe.get_feature_names_out(data_df.columns)

    # Convert the array into a readable DataFrame, preserving the original index
    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=data_df.index)
    return encoded_df

def manual_ordinal_encode(data, category_order):
    """
    Uses the Controlled Ordinal glyph.
    Maps categories to integers based on a user-defined order, preserving true meaning.

    Parameters:
    data (array-like): The categorical feature (e.g., a Series).
    category_order (list): The desired order of categories from low to high.
                          Example: ['low', 'medium', 'high']

    Returns:
    Series: A new Series with the ordinally encoded values.
    """
    # Create a mapping dictionary from the ordered list
    ordinal_mapping = {category: idx for idx, category in enumerate(category_order)}
    
    # Use .map() to apply the translation to the data
    return data.map(ordinal_mapping)

# Example Ritual Casting (for testing)
if __name__ == "__main__":
    print("🧙‍♂️ Beginning the Translation Ritual...\n")
    
    # Create a sample DataFrame with different types of categorical data
    df = pd.DataFrame({
        'country': ['USA', 'UK', 'FR', 'UK', 'USA', 'FR'],  # Nominal
        'size': ['small', 'large', 'medium', 'medium', 'small', 'large'], # Ordinal but unordered
        'priority': ['low', 'high', 'medium', 'low', 'high', 'medium']    # Clear Ordinal
    })
    
    print("Original Scroll (DataFrame):")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # 1. Demonstrate One-Hot Encoding on a NOMINAL feature: 'country'
    print("1. One-Hot Encoding the 'country' glyph (Nominal Data):")
    country_encoded = one_hot_encode(df['country'])
    print(country_encoded)
    print("\n" + "-"*30 + "\n")
    
    # 2. Demonstrate Manual Ordinal Encoding on an ORDINAL feature: 'priority'
    # We define the correct order for the categories
    priority_order = ['low', 'medium', 'high']
    print(f"2. Manual Ordinal Encoding the 'priority' glyph with order {priority_order}:")
    df['priority_encoded'] = manual_ordinal_encode(df['priority'], priority_order)
    print(df[['priority', 'priority_encoded']])
    print("\n" + "-"*30 + "\n")
    
    # 3. Demonstrate the DANGER of Label Encoding on a fake ordinal feature: 'size'
    print("3! WARNING: Label Encoding the 'size' glyph creates a FALSE ORDER:")
    df['size_dangerous_encoded'] = label_encode(df['size'])
    print(df[['size', 'size_dangerous_encoded']])
    print("    ⚠️  Notice: 'large'(2) > 'medium'(1) > 'small'(0) is correct...")
    
    # 4. But what if the original order was different? Show the problem.
    df_sizes_messy = pd.DataFrame({'size': ['large', 'small', 'medium']})
    df_sizes_messy['encoded'] = label_encode(df_sizes_messy['size'])
    print(f"    But if data arrives as {list(df_sizes_messy['size'])}:")
    print(f"    It becomes encoded as: {list(df_sizes_messy['encoded'])}")
    print("    ❌ This is wrong! 'large' should not be the smallest value. This is why we define order manually.")