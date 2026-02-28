"""
Data loading module for cardiac arrhythmia classification.
Handles ARFF file parsing and train/val/test splitting with stratification.
"""

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split


def load_arff(filepath):
    """
    Load ARFF dataset into pandas DataFrame.
    Convert '?' to NaN and decode byte strings.
    
    Parameters
    ----------
    filepath : str
        Path to .arff file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all features and target column (missing values as NaN)
    dict
        ARFF metadata
    """
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    
    # Decode byte strings to regular strings and convert '?' to NaN
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
            # Replace ARFF missing value marker with NaN
            df[col] = df[col].replace('?', np.nan)
    
    return df, meta


def extract_class_mapping(filepath):
    """
    Extract class descriptions from ARFF file comments.
    Parses the human-readable class descriptions from the file header.
    
    Parameters
    ----------
    filepath : str
        Path to .arff file
    
    Returns
    -------
    dict
        Mapping from class code (as string) to clinical description
        Example: {'1': 'Normal', '2': 'Ischemic changes...', ...}
    """
    class_mapping = {}
    
    try:
        with open(filepath, 'r') as f:
            in_class_section = False
            for line in f:
                # Look for the class code section marker
                if 'Class code :' in line:
                    in_class_section = True
                    continue
                
                if in_class_section:
                    # Stop at @RELATION line
                    if line.startswith('@RELATION'):
                        break
                    
                    # Parse class lines (format: "01  Normal  245")
                    line = line.strip()
                    if line and not line.startswith('%'):
                        continue
                    
                    if line.startswith('%'):
                        # Remove comment marker and trim
                        line = line[1:].strip()
                        
                        # Check if this is a class definition line
                        # Format: "01             Normal                            245"
                        if len(line) >= 2 and line[:2].isdigit():
                            parts = line.split()
                            if len(parts) >= 2:
                                class_code = str(int(parts[0]))  # Strip leading zeros
                                description = ' '.join(parts[1:-1]) if len(parts) > 2 else parts[1]
                                class_mapping[class_code] = description
    except Exception as e:
        print(f"Warning: Could not extract class mapping from {filepath}: {e}")
    
    return class_mapping


def handle_missing_values(df, strategy='drop', threshold=0.5):
    """
    Handle missing values in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    strategy : str, default='drop'
        Strategy for handling missing values:
        - 'drop': Remove rows with any missing values
        - 'mean': Impute numeric columns with mean
        - 'median': Impute numeric columns with median
    threshold : float, default=0.5
        Percentage threshold for dropping columns (0-1).
        Columns with > threshold missing values are dropped.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled
    """
    # Drop columns with too many missing values
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > threshold].index
    df = df.drop(columns=cols_to_drop)
    
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df


def split_data(df, target_col, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets with stratification.
    
    Uses 70% train/val (combined), 15% validation, 15% test split.
    Stratification is applied to preserve class distributions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str
        Name of target column
    test_size : float, default=0.15
        Proportion of data for test set
    val_size : float, default=0.15
        Proportion of train+val data for validation
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    train_X, train_y, val_X, val_y, test_X, test_y : tuple
        Train/val/test feature and target splits
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: test (15%) vs train+val (85%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # Second split: train vs val within train+val
    # val_size is proportion of temp data (train+val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        stratify=y_temp,
        random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def infer_target_column(df, preferred='Class'):
    """
    Infer target column with case-insensitive matching and fallback.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    preferred : str, default='Class'
        Preferred target column name

    Returns
    -------
    str
        Resolved target column name
    """
    if preferred in df.columns:
        return preferred

    lower_map = {col.lower(): col for col in df.columns}
    if preferred.lower() in lower_map:
        return lower_map[preferred.lower()]

    if 'class' in lower_map:
        return lower_map['class']

    return df.columns[-1]


def load_data(filepath, target_col='class', missing_value_strategy='drop', random_state=42):
    """
    Complete pipeline to load ARFF data and return train/val/test splits.
    Handles missing values, resolves target column, drops J feature.
    Also extracts class descriptions for interpretable labels.
    
    Parameters
    ----------
    filepath : str
        Path to .arff file
    target_col : str, default='class'
        Name of target column (use lowercase 'class')
    missing_value_strategy : str, default='drop'
        Strategy for handling missing values
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    train_X, train_y, val_X, val_y, test_X, test_y : tuple
        Train/val/test feature and target splits
    full_df : pd.DataFrame
        Full dataset (for reference and analysis)
    class_mapping : dict
        Mapping from class code (string) to clinical description
    """
    # Load data
    df, meta = load_arff(filepath)
    
    # Extract class descriptions
    class_mapping = extract_class_mapping(filepath)
    
    # Resolve target column robustly (case-insensitive)
    resolved_target_col = infer_target_column(df, preferred=target_col)
    
    print(f"  Resolved target column: '{resolved_target_col}'")
    print(f"  Initial shape: {df.shape}")
    
    # Handle missing values
    df_clean = handle_missing_values(df, strategy=missing_value_strategy)
    print(f"  After handling missing values: {df_clean.shape}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df_clean,
        target_col=resolved_target_col,
        random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, df_clean, class_mapping
