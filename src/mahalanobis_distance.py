import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve, pinvh


def mahalanobis_distance(U, V, regularization=1e-6, method='auto', use_shrinkage=True):
    """
    Compute Mahalanobis distance with automatic dimension detection
    
    Args:
        U: array-like - first dataset
        V: array-like - second dataset  
        regularization: regularization parameter for covariance matrix
        method: 'auto', 'pairwise', 'distribution', 'single'
        use_shrinkage: whether to use shrinkage covariance estimation
        
    Returns:
        Mahalanobis distance(s) - scalar, 1D array, or 2D array depending on method
        
    Method Selection Logic:
        - 'single': Single vector to vector distance
        - 'pairwise': Row-wise distances between corresponding rows  
        - 'distribution': Distance between two distributions (multivariate)
        - 'auto': Automatically select based on input dimensions
    """
    # Convert to numpy arrays
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    
    # Remove NaN values
    U_clean = U[~np.isnan(U).any(axis=-1)] if U.ndim > 1 else U[~np.isnan(U)]
    V_clean = V[~np.isnan(V).any(axis=-1)] if V.ndim > 1 else V[~np.isnan(V)]
    
    # Determine method automatically
    if method == 'auto':
        if U.ndim == 1 and V.ndim == 1:
            if len(U) == len(V):
                method = 'single'  # Two single vectors
            else:
                method = 'distribution'  # Different length vectors treated as distributions
        elif U.ndim == 2 and V.ndim == 2:
            if U.shape == V.shape and U.shape[0] > U.shape[1]:
                method = 'pairwise'
                # # Many samples, few features - likely want distribution comparison
                # method = 'distribution'
            elif U.shape == V.shape:
                # Same shape matrices - could be pairwise or distribution
                method = 'pairwise'
                # # Use distribution for health indicators (more meaningful)
                # method = 'distribution'
            else:
                method = 'distribution'  # Different shapes - distribution comparison
        else:
            method = 'distribution'  # Mixed dimensions
    
    # print(f"Using method: {method} for shapes U={U.shape}, V={V.shape}")
    
    if method == 'single':
        return _mahalanobis_single_vector(U_clean, V_clean, regularization, use_shrinkage)
    elif method == 'pairwise':
        return _mahalanobis_pairwise(U_clean, V_clean, regularization, use_shrinkage)
    elif method == 'distribution':
        return _mahalanobis_distribution(U_clean, V_clean, regularization, use_shrinkage)
    else:
        raise ValueError(f"Unknown method: {method}")


def _mahalanobis_single_vector(u, v, regularization=1e-6, use_shrinkage=True):
    """
    Distance between two single vectors using their combined covariance
    Most appropriate for: comparing two single feature vectors
    """
    if u.ndim > 1:
        u = u.ravel()
    if v.ndim > 1:
        v = v.ravel()
    
    if len(u) != len(v):
        raise ValueError(f"Single vectors must have same length: {len(u)} vs {len(v)}")
    
    # Use combined data to estimate covariance
    X_combined = np.vstack([u.reshape(1, -1), v.reshape(1, -1)])
    
    try:
        # Compute covariance matrix
        if use_shrinkage and X_combined.shape[0] > 2:
            try:
                from sklearn.covariance import LedoitWolf
                cov_matrix = LedoitWolf().fit(X_combined).covariance_
            except ImportError:
                cov_matrix = np.cov(X_combined, rowvar=False)
        else:
            cov_matrix = np.cov(X_combined, rowvar=False)
        
        # Add regularization
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[cov_matrix + regularization]])
        else:
            cov_matrix += np.eye(cov_matrix.shape[0]) * regularization
        
        # Compute distance
        diff = u - v
        inv_cov = np.linalg.pinv(cov_matrix)
        distance = np.sqrt(diff.T @ inv_cov @ diff)
        
        return float(distance)
        
    except Exception as e:
        # Fallback to Euclidean distance
        return float(np.linalg.norm(u - v))


def _mahalanobis_pairwise(U, V, regularization=1e-6, use_shrinkage=True):
    """
    Row-wise distances between corresponding rows of two matrices
    Most appropriate for: comparing reconstructions point-by-point
    """
    if U.shape != V.shape:
        raise ValueError(f"Matrices must have same shape: {U.shape} vs {V.shape}")
    
    n_samples, n_features = U.shape
    
    # Compute combined covariance matrix
    X_combined = np.vstack([U, V])
    
    try:
        if use_shrinkage and X_combined.shape[0] > 2:
            try:
                from sklearn.covariance import LedoitWolf
                cov_matrix = LedoitWolf().fit(X_combined).covariance_
            except ImportError:
                cov_matrix = np.cov(X_combined, rowvar=False)
        else:
            cov_matrix = np.cov(X_combined, rowvar=False)
        
        # Add regularization
        cov_matrix += np.eye(n_features) * regularization
        inv_cov = np.linalg.pinv(cov_matrix)
        
        # Compute distances for each row pair
        distances = []
        for i in range(n_samples):
            diff = U[i] - V[i]
            distance = np.sqrt(diff.T @ inv_cov @ diff)
            distances.append(float(distance))
        
        return np.array(distances)
        
    except Exception:
        # Fallback to Euclidean distances
        return np.linalg.norm(U - V, axis=1)


def _mahalanobis_distribution(U, V, regularization=1e-6, use_shrinkage=True):
    """
    Distance between two multivariate distributions (most appropriate for health indicators)
    
    This computes the Mahalanobis distance between the centroids of two distributions,
    using the pooled covariance matrix. This is most meaningful for health indicators
    because it measures how different the two distributions are in the space defined
    by their combined variability.
    
    Most appropriate for: comparing original vs reconstructed feature distributions
    """
    # Ensure 2D arrays
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    
    # Compute means (centroids)
    mean_U = np.mean(U, axis=0)
    mean_V = np.mean(V, axis=0)
    
    if U.shape[1] != V.shape[1]:
        raise ValueError(f"Feature dimensions must match: {U.shape[1]} vs {V.shape[1]}")
    
    # Compute pooled covariance matrix
    try:
        if use_shrinkage and (U.shape[0] + V.shape[0]) > U.shape[1] + 2:
            try:
                from sklearn.covariance import LedoitWolf
                # Fit on combined data for pooled covariance
                X_combined = np.vstack([U, V])
                cov_matrix = LedoitWolf().fit(X_combined).covariance_
            except ImportError:
                cov_U = np.cov(U, rowvar=False) if U.shape[0] > 1 else np.zeros((U.shape[1], U.shape[1]))
                cov_V = np.cov(V, rowvar=False) if V.shape[0] > 1 else np.zeros((V.shape[1], V.shape[1]))
                # Pooled covariance
                n_U, n_V = U.shape[0], V.shape[0]
                cov_matrix = ((n_U - 1) * cov_U + (n_V - 1) * cov_V) / (n_U + n_V - 2)
        else:
            cov_U = np.cov(U, rowvar=False) if U.shape[0] > 1 else np.zeros((U.shape[1], U.shape[1]))
            cov_V = np.cov(V, rowvar=False) if V.shape[0] > 1 else np.zeros((V.shape[1], V.shape[1]))
            n_U, n_V = U.shape[0], V.shape[0]
            cov_matrix = ((n_U - 1) * cov_U + (n_V - 1) * cov_V) / max(n_U + n_V - 2, 1)
        
        # Add regularization
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[cov_matrix + regularization]])
        else:
            cov_matrix += np.eye(cov_matrix.shape[0]) * regularization
        
        # Compute Mahalanobis distance between centroids
        diff = mean_U - mean_V
        inv_cov = np.linalg.pinv(cov_matrix)
        distance = np.sqrt(diff.T @ inv_cov @ diff)
        
        return float(distance)
        
    except Exception as e:
        # Fallback to Euclidean distance between means
        return float(np.linalg.norm(mean_U - mean_V))


def mahalanobis_distance_health_indicator(original_features, reconstructed_features, regularization=1e-6):
    """
    Specialized function for health indicator generation
    
    Computes per-sample Mahalanobis distances in the space defined by the 
    combined covariance of original and reconstructed features.
    
    This gives a health indicator value for each time sample.
    
    Args:
        original_features: (n_samples, n_features) - original feature matrix
        reconstructed_features: (n_samples, n_features) - reconstructed feature matrix
        
    Returns:
        health_indicator: (n_samples,) - Mahalanobis distance for each sample
    """
    original_features = np.asarray(original_features, dtype=np.float64)
    reconstructed_features = np.asarray(reconstructed_features, dtype=np.float64)
    
    if original_features.shape != reconstructed_features.shape:
        raise ValueError(f"Shape mismatch: {original_features.shape} vs {reconstructed_features.shape}")
    
    n_samples, n_features = original_features.shape
    
    # Compute pooled covariance from all data
    X_combined = np.vstack([original_features, reconstructed_features])
    
    # Use shrinkage covariance if available and beneficial
    try:
        from sklearn.covariance import LedoitWolf
        if X_combined.shape[0] > n_features + 10:  # Sufficient samples for shrinkage
            cov_estimator = LedoitWolf()
            cov_matrix = cov_estimator.fit(X_combined).covariance_
        else:
            cov_matrix = np.cov(X_combined, rowvar=False)
    except ImportError:
        cov_matrix = np.cov(X_combined, rowvar=False)
    
    # Add regularization for numerical stability
    cov_matrix += np.eye(n_features) * regularization
    
    # Compute inverse
    inv_cov = np.linalg.pinv(cov_matrix)
    
    # Compute per-sample Mahalanobis distances
    health_indicator = []
    for i in range(n_samples):
        diff = original_features[i] - reconstructed_features[i]
        distance = np.sqrt(diff.T @ inv_cov @ diff)
        health_indicator.append(distance)
    
    return np.array(health_indicator)
        

# Legacy compatibility
def mahalanobis_distance_single(u, v, X_ref=None, reg=1e-6, max_reg=1e-1, use_shrinkage=True):
    """Legacy single vector function - kept for backward compatibility"""
    return _mahalanobis_single_vector(u, v, reg, use_shrinkage)
