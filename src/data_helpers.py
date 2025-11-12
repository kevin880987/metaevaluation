

import os
import pandas as pd
import numpy as np
import pickle


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def swap_dict_level(dict_):
    new_dict = {}
    for a, nested in dict_.items():
        for b, val in nested.items():
            new_dict.setdefault(b, dict())[a] = val
    for a, nested in new_dict.items():
        new_dict[a] = pd.DataFrame.from_dict(nested)
    
    return new_dict


# def swap_dict_level(dict_):
#     """
#     Swap dictionary levels without converting to DataFrame
#     Preserves original data structure and lengths per experiment
    
#     From: {experiment: {health_indicator: values}}
#     To:   {health_indicator: {experiment: values}}
#     """
#     new_dict = {}
#     for experiment, health_indicators in dict_.items():
#         for health_indicator, values in health_indicators.items():
#             new_dict.setdefault(health_indicator, {})[experiment] = values
    
#     return new_dict


# def split_data(holder, test_ratio):
#     def split(dfs, test_ratio):
#         output = []
#         for df in dfs:
#             test = int(df.shape[0]*test_ratio)
#             output.extend([df.iloc[:-test], df.iloc[-test:]])
#         return output

#     X_dict = holder.item['X']
#     E_dict = holder.item['E']
#     X_train, X_test, E_train, E_test, X, E = [], [], [], [], [], []
#     for experiment, df in X_dict.items():
#         x = df
#         e = pd.get_dummies(pd.DataFrame.from_dict(E_dict[experiment]))
#         x_train, x_test, e_train, e_test = split([x, e], test_ratio)
#         X_train.append(x_train)
#         X_test.append(x_test)
#         E_train.append(e_train)
#         E_test.append(e_test)
#         X.append(x)
#         E.append(e)

#     X_train, X_test, E_train, E_test, X, E = [pd.concat(dfs).reset_index(drop=True) \
#                                         for dfs in [X_train, X_test, E_train, E_test, X, E]]
#     E_train, E_test, E = E_train.fillna(False), E_test.fillna(False), E.fillna(False)
#     return X_train, X_test, E_train, E_test, X, E


def stack_sequence(df, in_seq, out_seq, idx=None, format_2d=True):
    """
    Stack sequences and return in 2D format by default, 3D for PyTorch compatibility

    Args:
        df: DataFrame with features
        in_seq: Input sequence positions 
        out_seq: Output sequence positions
        idx: Indices to use
        format_2d: If True, return 2D format [samples, features * sequence]
                    If False, return 3D format [samples, features, sequence]
    
    Returns:
        3D numpy array or 2D DataFrame depending on return_3d parameter
    """
    if idx is None:
        idx = df.index
        
    def _get_sequence(s):
        sequence = df.shift(-s).loc[idx]
        if s > 0:
            sequence.columns = [f'{c}_t+{s}' for c in sequence.columns]
        elif s < 0:
            sequence.columns = [f'{c}_t{s}' for c in sequence.columns]
        else:
            sequence.columns = [f'{c}' for c in sequence.columns]
        return sequence
        
    seq = sorted(list(set([-s for s in in_seq] + out_seq)))
    stacked_df = pd.concat([_get_sequence(s) for s in seq], axis=1)
    stacked_df = stacked_df.dropna(axis=0)
    
    if format_2d:
        # Return 2D DataFrame for sklearn compatibility
        return stacked_df
    else:
        # Convert to 3D format: [samples, features, sequence_length]
        n_features = len(df.columns)
        sequence_length = len(seq)
        
        # Convert to 3D numpy array
        X_3d = sklearn_to_pytorch_format(
            stacked_df.values, 
            n_features=n_features, 
            sequence_length=sequence_length
        )
        return X_3d


def sklearn_to_pytorch_format(X_2d, n_features, sequence_length):
    """
    Transform 2D sklearn format to 3D PyTorch format for sequence data
    
    Args:
        X_2d: 2D array/DataFrame of shape [n_samples, n_features * sequence_length]
              Features are arranged as [feat1_t0, feat1_t1, ..., feat1_tN, feat2_t0, ...]
        n_features: Number of original features (before sequence expansion)
        sequence_length: Length of input sequence
        
    Returns:
        X_3d: 3D array of shape [n_samples, n_features, sequence_length]
              Suitable for PyTorch models expecting temporal architecture
    """
    # Convert to numpy array if it's a DataFrame
    if isinstance(X_2d, pd.DataFrame):
        X_array = X_2d.values
    else:
        X_array = np.array(X_2d)
    
    n_samples = X_array.shape[0]
    
    # Verify input dimensions
    expected_columns = n_features * sequence_length
    if X_array.shape[1] != expected_columns:
        raise ValueError(f"Expected {expected_columns} columns (n_features={n_features} * sequence_length={sequence_length}), "
                        f"but got {X_array.shape[1]} columns")
    
    # Reshape: [n_samples, n_features * sequence_length] -> [n_samples, n_features, sequence_length]
    X_3d = X_array.reshape(n_samples, n_features, sequence_length)
    
    return X_3d


# def pytorch_to_sklearn_format(X_3d):
#     """
#     Transform 3D PyTorch format back to 2D sklearn format
    
#     Args:
#         X_3d: 3D array of shape [n_samples, n_features, sequence_length]
        
#     Returns:
#         X_2d: 2D array of shape [n_samples, n_features * sequence_length]
#               Features arranged as [feat1_t0, feat1_t1, ..., feat1_tN, feat2_t0, ...]
#     """
#     X_array = np.array(X_3d)
#     n_samples, n_features, sequence_length = X_array.shape
    
#     # Reshape: [n_samples, n_features, sequence_length] -> [n_samples, n_features * sequence_length]
#     X_2d = X_array.reshape(n_samples, n_features * sequence_length)
    
#     return X_2d


# def get_feature_names_for_sequence(original_features, sequence_positions):
#     """
#     Generate feature names for sequence data in sklearn format
    
#     Args:
#         original_features: List of original feature names
#         sequence_positions: List of sequence positions (e.g., [0, 1, 2] for 3-step sequence)
        
#     Returns:
#         List of feature names in sklearn format [feat1_t0, feat1_t1, ..., feat1_tN, feat2_t0, ...]
#     """
#     feature_names = []
#     for feature in original_features:
#         for pos in sequence_positions:
#             if pos == 0:
#                 feature_names.append(feature)
#             elif pos > 0:
#                 feature_names.append(f"{feature}_t+{pos}")
#             else:
#                 feature_names.append(f"{feature}_t({pos})")

#     return feature_names

