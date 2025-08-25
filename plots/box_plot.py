# boxplot.py - The Seer's Looking Glass for Outliers

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def create_boxplot(data, x=None, y=None, hue=None, title="Boxplot", 
                   xlabel="", ylabel="Values", orient="v", palette="Set2",
                   show_outliers=True, show_means=False, figsize=(10, 6)):
    """
    Creates a boxplot to visualize data distribution and identify outliers.
    
    Parameters:
    data : DataFrame, array, or list
        The data to visualize
    x : str, optional
        Variable for x-axis (for categorical grouping)
    y : str, optional
        Variable for y-axis (for numerical values)
    hue : str, optional
        Additional categorical grouping variable
    title : str
        Title of the plot
    xlabel, ylabel : str
        Labels for x and y axes
    orient : str
        Orientation: 'v' for vertical, 'h' for horizontal
    palette : str
        Color palette for the plot
    show_outliers : bool
        Whether to show outliers as points
    show_means : bool
        Whether to show mean markers
    figsize : tuple
        Figure size (width, height)
    
    """
    
    # Set the visual style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=figsize)
    
    # Create the boxplot
    if isinstance(data, pd.DataFrame) and x and y:
        # DataFrame with x and y specified
        ax = sns.boxplot(data=data, x=x, y=y, hue=hue, palette=palette, 
                        orient=orient, showfliers=show_outliers)
        
        if show_means:
            # Add mean markers
            means = data.groupby(x)[y].mean().values
            for i, mean in enumerate(means):
                ax.scatter(i, mean, marker='o', color='red', s=100, zorder=10, label='Mean' if i == 0 else "")
                
    elif isinstance(data, pd.Series) or isinstance(data, list) or isinstance(data, np.ndarray):
        # Single dimension data
        ax = sns.boxplot(data=data, orient=orient, palette=palette, showfliers=show_outliers)
        
        if show_means:
            # Add mean marker
            mean_val = np.mean(data)
            if orient == "v":
                ax.scatter(0, mean_val, marker='o', color='red', s=100, zorder=10, label='Mean')
            else:
                ax.scatter(mean_val, 0, marker='o', color='red', s=100, zorder=10, label='Mean')
    else:
        # DataFrame without x and y (plot all numerical columns)
        numerical_data = data.select_dtypes(include=[np.number])
        ax = sns.boxplot(data=numerical_data, orient=orient, palette=palette, showfliers=show_outliers)
        
        if show_means:
            # Add mean markers for each column
            means = numerical_data.mean().values
            for i, mean in enumerate(means):
                if orient == "v":
                    ax.scatter(i, mean, marker='o', color='red', s=100, zorder=10, label='Mean' if i == 0 else "")
                else:
                    ax.scatter(mean, i, marker='o', color='red', s=100, zorder=10, label='Mean' if i == 0 else "")
    
    # Customize the plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Add legend if means are shown
    if show_means:
        handles, labels = ax.get_legend_handles_labels()
        if labels:  # Only add legend if there are labels
            # Remove duplicate labels
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels if they're long
    if x and isinstance(data, pd.DataFrame):
        if len(data[x].unique()) > 5:
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    return ax

def interpret_boxplot(ax):
    """
    Provides a basic interpretation of the boxplot elements.
    
    Parameters:
    ax : matplotlib axis
        The axis containing the boxplot
    """
    print("\n🔍 Interpretation of the Boxplot:")
    print("• The central line in each box represents the median (50th percentile)")
    print("• The box extends from Q1 (25th percentile) to Q3 (75th percentile)")
    print("• The whiskers typically extend to 1.5 * IQR (Interquartile Range)")
    print("• Points beyond the whiskers are potential outliers")
    print("• A wide box indicates high variability in the data")
    print("• Asymmetric boxes suggest skewness in the distribution")

# Example usage
if __name__ == "__main__":
    print("🧙‍♂️ Gazing into the Looking Glass...")
    
    # Create sample data with outliers
    np.random.seed(42)
    data = {
        'Category': ['A'] * 50 + ['B'] * 50 + ['C'] * 50,
        'Values': np.concatenate([
            np.random.normal(50, 10, 50),  # Category A
            np.random.normal(80, 15, 50),  # Category B
            np.random.normal(30, 5, 50)    # Category C
        ])
    }
    
    # Add some outliers
    data['Values'][10] = 120  # Outlier in Category A
    data['Values'][60] = 150  # Outlier in Category B
    data['Values'][110] = 5   # Outlier in Category C
    
    df = pd.DataFrame(data)
    
    # Create a boxplot
    ax = create_boxplot(data=df, x='Category', y='Values', 
                       title="Distribution of Values by Category",
                       ylabel="Measurement Values", show_means=True)
    
    # Add interpretation
    interpret_boxplot(ax)
    
    # Show the plot
    plt.show()
    
    # Demonstrate with a single array
    print("\n" + "="*50)
    print("Gazing at a single numerical array...")
    single_data = np.random.normal(100, 20, 200)
    single_data[0] = 250  # Add an extreme outlier
    
    ax2 = create_boxplot(single_data, title="Distribution of Single Variable", 
                        ylabel="Values", orient="h")
    interpret_boxplot(ax2)
    plt.show()