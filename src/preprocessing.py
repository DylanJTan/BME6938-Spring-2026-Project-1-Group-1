"""
Preprocessing module for data preparation and feature engineering.
Handles feature scaling, categorical encoding, and data normalization.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def drop_j_feature(X):
    """
    Drop J feature (column 13, 0-indexed, named "'J'" with quotes).
    J point is instantaneous location, not a measurable wave vector.
    Clinical decision: J should not be used for arrhythmia classification.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    
    Returns
    -------
    pd.DataFrame
        Feature matrix with J column removed
    """
    # J feature is at position 13 (0-indexed) and named "'J'" (with quotes)
    columns_to_try = ["'J'", 'J', 'attr_14']
    
    # Try to drop by exact name match
    for col_name in columns_to_try:
        if col_name in X.columns:
            X = X.drop(columns=[col_name])
            return X
    
    # If not found by name, try position-based drop (column 13)
    if X.shape[1] > 13:
        col_to_drop = X.columns[13]
        if 'J' in str(col_to_drop).upper():
            X = X.drop(columns=[col_to_drop])
    
    return X


def identify_feature_types(X):
    """
    Identify numeric and categorical features.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input features
    
    Returns
    -------
    numeric_features : list
        List of numeric column names
    categorical_features : list
        List of categorical column names
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    return numeric_features, categorical_features


def encode_categorical_features(X_train, X_val, X_test, drop_j=True):
    """
    Encode categorical features using LabelEncoder.
    Fit on training data, apply to all splits.
    Optionally drop J feature before encoding.
    
    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Train, validation, and test feature sets
    drop_j : bool, default=True
        Whether to drop J feature (attr_14) before encoding
    
    Returns
    -------
    X_train_enc, X_val_enc, X_test_enc : pd.DataFrame
        Encoded feature sets
    encoders : dict
        Dictionary mapping column names to category->index maps
    """
    # Drop J feature if present
    if drop_j:
        X_train = drop_j_feature(X_train)
        X_val = drop_j_feature(X_val)
        X_test = drop_j_feature(X_test)
    
    _, categorical_features = identify_feature_types(X_train)
    
    if not categorical_features:
        return X_train.copy(), X_val.copy(), X_test.copy(), {}
    
    X_train_enc = X_train.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy()
    encoders = {}
    
    for col in categorical_features:
        train_values = X_train[col].astype(str)
        categories = sorted(train_values.unique())
        category_map = {value: index for index, value in enumerate(categories)}

        X_train_enc[col] = train_values.map(category_map).astype(int)
        X_val_enc[col] = X_val[col].astype(str).map(category_map).fillna(-1).astype(int)
        X_test_enc[col] = X_test[col].astype(str).map(category_map).fillna(-1).astype(int)
        encoders[col] = category_map
    
    return X_train_enc, X_val_enc, X_test_enc, encoders


def scale_features(X_train, X_val, X_test, scaler_type='standard'):
    """
    Scale numeric features using specified scaler.
    Fit on training data, apply to all splits.
    
    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame or ndarray
        Train, validation, and test feature sets
    scaler_type : str, default='standard'
        Type of scaler: 'standard' (StandardScaler)
    
    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled : pd.DataFrame or ndarray
        Scaled feature sets
    scaler : sklearn scaler object
        Fitted scaler for future use
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Fit on train data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Apply to val and test
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Return as DataFrame if input was DataFrame
    if isinstance(X_train, pd.DataFrame):
        columns = X_train.columns
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=columns)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def preprocess_data(X_train, X_val, X_test, scale=True, encode_categorical=True, drop_j=True):
    """
    Complete preprocessing pipeline: drop J feature, categorical encoding, feature scaling.
    
    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Train, validation, and test feature sets
    scale : bool, default=True
        Whether to apply feature scaling
    encode_categorical : bool, default=True
        Whether to encode categorical features
    drop_j : bool, default=True
        Whether to drop J feature (attr_14) before encoding
    
    Returns
    -------
    X_train_prep, X_val_prep, X_test_prep : ndarray or pd.DataFrame
        Preprocessed feature sets
    preprocessing_info : dict
        Dictionary containing preprocessing metadata (encoders, scaler)
    """
    X_train_prep = X_train.copy()
    X_val_prep = X_val.copy()
    X_test_prep = X_test.copy()
    preprocessing_info = {}
    
    # Encode categorical features (includes J dropping if enabled)
    if encode_categorical:
        X_train_prep, X_val_prep, X_test_prep, encoders = encode_categorical_features(
            X_train_prep, X_val_prep, X_test_prep, drop_j=drop_j
        )
        preprocessing_info['encoders'] = encoders
    
    # Scale features
    if scale:
        X_train_prep, X_val_prep, X_test_prep, scaler = scale_features(
            X_train_prep, X_val_prep, X_test_prep, scaler_type='standard'
        )
        preprocessing_info['scaler'] = scaler
    
    return X_train_prep, X_val_prep, X_test_prep, preprocessing_info


def select_features_by_variance(X_train, X_val, X_test, threshold=0.0):
    """
    Select features based on variance threshold to remove low-variance features.
    
    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame or ndarray
        Train, validation, and test feature sets
    threshold : float, default=0.0
        Variance threshold below which features are removed
    
    Returns
    -------
    X_train_sel, X_val_sel, X_test_sel : pd.DataFrame or ndarray
        Feature-selected datasets
    selected_features : list
        List of selected feature names or indices
    """
    from sklearn.feature_selection import VarianceThreshold
    
    selector = VarianceThreshold(threshold=threshold)
    X_train_sel = selector.fit_transform(X_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    
    # Get selected feature names if DataFrame
    if isinstance(X_train, pd.DataFrame):
        selected_features = X_train.columns[selector.get_support()].tolist()
        X_train_sel = pd.DataFrame(X_train_sel, columns=selected_features)
        X_val_sel = pd.DataFrame(X_val_sel, columns=selected_features)
        X_test_sel = pd.DataFrame(X_test_sel, columns=selected_features)
    else:
        selected_features = selector.get_support(indices=True).tolist()
    
    return X_train_sel, X_val_sel, X_test_sel, selected_features


def encode_target(y_train, y_val, y_test):
    """
    Encode target variable using LabelEncoder.
    
    Parameters
    ----------
    y_train, y_val, y_test : pd.Series or ndarray
        Train, validation, and test targets
    
    Returns
    -------
    y_train_enc, y_val_enc, y_test_enc : ndarray
        Encoded targets
    label_encoder : LabelEncoder
        Fitted encoder for decoding predictions
    """
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train.astype(str))
    y_val_enc = le.transform(y_val.astype(str))
    y_test_enc = le.transform(y_test.astype(str))
    
    return y_train_enc, y_val_enc, y_test_enc, le
