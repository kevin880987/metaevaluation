

import numpy as np
import pandas as pd
from numpy.linalg import norm
from itertools import combinations, product
from scipy.interpolate import make_interp_spline, BSpline
import pymannkendall as mk
from mahalanobis_distance import mahalanobis_distance
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_helpers import swap_dict_level
# from consistency import get_consistency, select_feature


# def to_probability_distribution(data, bins=50):
#     counts, bin_edges = np.histogram(data, bins=bins, density=True)
#     bin_widths = np.diff(bin_edges)
#     distribution = counts * bin_widths
#     return distribution
    

# def jensenshannon_divergence(P, Q):
#     def calculate(P, Q):
#         # Normalize to distribution that sum to 1
#         if np.nanmax(P)!=np.nanmin(P):
#             P -= np.nanmin(P)
#         if np.nanmax(Q)!=np.nanmin(Q):
#             Q -= np.nanmin(Q)
#         P /= np.nansum(P)
#         Q /= np.nansum(Q)
#         valid = np.all((P>0, Q>0, P==P, Q==Q), axis=0)
#         P, Q = P[valid], Q[valid]

#         # Claculate
#         _P = P / norm(P, ord=1)
#         _Q = Q / norm(Q, ord=1)
#         _M = 0.5 * (_P + _Q)
#         entropy = lambda P, Q: np.nansum(P*np.log2(P/Q, where=P/Q>0))
#         divergence = (0.5*(entropy(_P, _M) + entropy(_Q, _M)))#**2
#         return divergence
    
#     P, Q = np.array(P), np.array(Q)
#     assert P.shape==Q.shape
#     if P.ndim==1:
#         divergence = calculate(P, Q)
#     else:
#         P, Q = np.reshape(P, (P.shape[0], -1)), np.reshape(Q, (Q.shape[0], -1))
#         divergence = []
#         for p, q in zip(P.T, Q.T):
#             divergence.append(calculate(p, q))

#     return divergence


# def get_scalability(X, Y):
#     pass


def get_internal_robustness(X, degree=2):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
    # https://www.youtube.com/watch?v=JwN43QAlF50&list=RDCMUC4r_gpcPz3Us9pmeGDjw_Rg&index=7&ab_channel=Graphicsin5Minutes
    # def _cal(x):
    #     x = x.dropna().drop_duplicates()
    #     # y = y.loc[x.index]
    #     sort = x.reset_index(drop=True).sort_values().index
    #     x = x.iloc[sort].values
    #     # y = y.iloc[sort].values
    #     b = make_interp_spline(np.arange(x.size), x)
    #     return -((np.nanmean((x - b(np.arange(x.size))) ** 2)) ** 0.5)
    def _cal(x):
        x = StandardScaler().fit_transform(x.dropna().to_frame())
        # Define knot points (evenly spaced for simplicity)
        knots = np.linspace(0, len(x) - 1, len(x) + degree + 1)
        # Fit the B-spline
        spline = BSpline(t=knots, c=x, k=degree)
        # Calculate mean squared difference
        squared_diff = np.nanmean((x - spline(np.arange(len(x))))**2)
        return -np.log(squared_diff)
    
    result = [_cal(x) for (_, x) in X.items()]
    return result


def get_external_robustness(X, Y):
    if X.shape[1]<2:
        return np.nan
    
    def _cal(sub1, sub2):
        sub1, sub2 = sub1[1].dropna().values, sub2[1].dropna().values
        n_samples = min(len(sub1), len(sub2))
        # shift = min(min(sub1), min(sub2))
        # sub1 -= shift
        # sub2 -= shift
        return -1*(jensenshannon_divergence(
            # sub1[np.linspace(0, len(sub1)-1, num=n_samples, dtype=int)], 
            # sub2[np.linspace(0, len(sub2)-1, num=n_samples, dtype=int)], 
            to_probability_distribution(sub1[-n_samples:], int(n_samples/50)), 
            to_probability_distribution(sub2[-n_samples:], int(n_samples/50)), 
            ))

    subtraction = Y - X
    result = [_cal(sub1, sub2) for sub1, sub2 in combinations(subtraction.items(), 2)]
    return result


# def get_reliability(X, Y):
#     def _cal(x, y):
#         x = x.dropna()
#         y = y.loc[x.index]
#         x_diff1 = x.diff().replace(0, np.nan).dropna()
#         x_diff2 = x_diff1.diff().replace(0, np.nan).dropna()
#         diff1 = np.nanmean(jensenshannon_divergence(
#             to_probability_distribution(x_diff1, int(x_diff1.size/50)), 
#             to_probability_distribution(y.loc[x_diff1.index], int(x_diff1.size/50)), 
#             ))
#         diff2 = np.nanmean(jensenshannon_divergence(
#             P=to_probability_distribution(x_diff2, int(x_diff2.size/50)), 
#             Q=to_probability_distribution(y.loc[x_diff2.index], int(x_diff2.size/50)), 
#             ))
#         return [-1*diff1, -1*diff2]
#     result = np.sum([_cal(x, Y[hi]) for (hi, x) in X.items()], axis=1)
#     return result


def _znorm(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    return (x - mu) / (sd + 1e-12)

def _softmin3(a, b, c, gamma):
    # numerically stable soft-min over three values
    m = min(a, b, c)
    return m - gamma * np.log(
        np.exp((m - a) / gamma) + np.exp((m - b) / gamma) + np.exp((m - c) / gamma)
    )

def soft_dtw(x, y, gamma=0.05, band=None):
    """
    Soft-DTW distance between 1D series x and y (z-normalization not included).
    Args:
        x, y : 1D arrays
        gamma: float > 0, smoothing temperature
        band : None or int, Sakoe–Chiba band half-width (|i-j| <= band)
    Returns:
        sdtw distance (float)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return np.nan

    # squared-Euclidean local costs
    D = (x[:, None] - y[None, :]) ** 2

    INF = 1e18
    R = np.full((n + 1, m + 1), INF, dtype=float)
    R[0, 0] = 0.0

    if band is None:
        band = max(n, m)  # unconstrained
    band = int(band)

    for i in range(1, n + 1):
        j_lo = max(1, i - band)
        j_hi = min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            R[i, j] = D[i - 1, j - 1] + _softmin3(R[i - 1, j], R[i, j - 1], R[i - 1, j - 1], gamma)
        # everything outside the band in row i stays INF

    return R[n, m]

def soft_dtw_normalized(x, y, gamma=0.05, band_ratio=0.10):
    """
    Z-normalize series, compute Soft-DTW with a Sakoe–Chiba band, and length-normalize.
    """
    xz = _znorm(x); yz = _znorm(y)
    T = max(len(xz), len(yz))
    band = int(band_ratio * T)
    dist = soft_dtw(xz, yz, gamma=gamma, band=band)
    # length-normalize by approximate path length (n + m)
    L = len(xz) + len(yz)
    return dist / max(L, 1)

def soft_dtw_derivative_normalized(x, y, gamma=0.05, band_ratio=0.10):
    """
    Derivative variant (DDTW): apply Soft-DTW to first differences of z-normalized series.
    """
    xz = _znorm(x); yz = _znorm(y)
    dx, dy = np.diff(xz), np.diff(yz)
    if len(dx) == 0 or len(dy) == 0:
        return np.nan
    T = max(len(dx), len(dy))
    band = int(band_ratio * T)
    dist = soft_dtw(dx, dy, gamma=gamma, band=band)
    L = len(dx) + len(dy)
    return dist / max(L, 1)

def reliability_soft_dtw(hi_series_dict, rul_series_dict, gamma=0.05, band_ratio=0.10, use_ddtw=True):
    """
    Compute Reliability scores for multiple runs using Soft-DTW (and optional DDTW).
    Args:
        hi_series_dict  : dict run_id -> pandas.Series (HI)
        rul_series_dict : dict run_id -> pandas.Series (RUL)
    Returns:
        dict run_id -> raw distance, and helper to min-max scale later
    """
    dists = {}
    for (rid, x), (_, y) in zip(hi_series_dict.items(), rul_series_dict.items()):
        # align by intersection of time indices
        x = x.dropna()
        y = y.dropna()
        idx = x.index.intersection(y.index)
        xv = x.loc[idx].values
        yv = y.loc[idx].values
        if len(xv) < 10 or len(yv) < 10:
            dists[rid] = np.nan
            continue
        d1 = soft_dtw_normalized(xv, yv, gamma=gamma, band_ratio=band_ratio)
        if use_ddtw:
            d2 = soft_dtw_derivative_normalized(xv, yv, gamma=gamma, band_ratio=band_ratio)
            d = np.nanmean([d1, d2])
        else:
            d = d1
        dists[rid] = d
    return dists

# def minmax_to_score(dist_dict):
#     """
#     Convert distances to [0,100] scores: score = 100 * (max - d) / (max - min).
#     """
#     vals = np.array([v for v in dist_dict.values() if np.isfinite(v)])
#     if len(vals) == 0:
#         return {k: np.nan for k in dist_dict}
#     dmin, dmax = np.min(vals), np.max(vals)
#     rng = dmax - dmin if dmax > dmin else 1.0
#     return {k: 100.0 * (dmax - v) / rng if np.isfinite(v) else np.nan for k, v in dist_dict.items()}

def get_reliability(X, Y, band_ratio=0.1, use_ddtw=False):
    """
    DTW-based reliability: higher is better.
    X: dict-like of HIs (per run)   ; Y: dict-like of RUL (per run)
    # https://github.com/mblondel/soft-dtw 
    # https://proceedings.mlr.press/v70/cuturi17a.html?source=post_page628e4799533c
    """
    d_raw = reliability_soft_dtw(X, Y, gamma=0.05, band_ratio=band_ratio, use_ddtw=use_ddtw)
    # reliability_scores = minmax_to_score(d_raw)  # higher = better, in [0,100]
    return list(d_raw.values())


def get_monotonicity(X):
    # def _cal(x):
    #     x = x.dropna()
    #     return np.abs((x>=0).sum()/(x.shape[0]-1)-(x<0).sum()/(x.shape[0]-1))
    # result = [_cal(x) for _, x in X.diff().items()]

    # https://github.com/mmhs013/pyMannKendall
    def _cal(x):
        x = x.dropna()
        return np.abs(mk.original_test(x, alpha=.05).Tau)
    result = [_cal(x) for _, x in X.items()]

    return result


def get_prognosability(X):
    failure_values = np.array([x.dropna().iloc[-1] for _, x in X.items()])
    starting_values = np.array([x.dropna().iloc[0] for _, x in X.items()])
    result = np.exp(np.std(X, axis=0).values/np.nanmean(np.abs(failure_values-starting_values)))
    return result


def get_trendability(X, Y):
    def _cal(x, y):
        x = x.dropna()
        y = y.loc[x.index]
        return np.abs(np.corrcoef(x, y)).min()
    result = [_cal(x, Y[hi]) for (hi, x) in X.items()]
    return result


def score(hi_dict, rul_dict):
    # Swap from {experiment: {health indicator: values}} to {health indicator: {experiment: values}}
    hi_dict = swap_dict_level(hi_dict)
    rul_dict = swap_dict_level(rul_dict)

    # Get scores of health indicator
    score_df = pd.DataFrame(index=hi_dict.keys())
    for (hi, X), (rul, Y) in product(hi_dict.items(), rul_dict.items()):
        # if hi=='REC MD': break
        if np.any(X.nunique()<2): continue
        
        score_df.loc[hi, "Monotonicity"] = np.nanmean(get_monotonicity(X))
        score_df.loc[hi, "Prognosability"] = np.nanmean(get_prognosability(X))
        score_df.loc[hi, "Trendability"] = np.nanmean(get_trendability(X, Y))
        
        score_df.loc[hi, "Reliability"] = np.nanmean(get_reliability(X, Y))
        # score_df.loc[hi, "External Robustness"] = np.nanmean(get_external_robustness(X, Y))
        score_df.loc[hi, "Internal Robustness"] = np.nanmean(get_internal_robustness(X))

    return score_df

    # Prepare for consistency
    # # Feature selection
    # selection = pd.DataFrame()
    # for _, X in holder.get("X").groupby(level=["Y", "Domain"], axis=1):
    #     Y = holder.get("Y").loc[:, X.columns[[0]]].droplevel(["X"], axis=1)
    #     selection = pd.concat([selection, select_feature(X, Y)], axis=1)
    # selection = selection.loc[:, holder.signals]

    # # Others
    # score_dict = {}
    # for (signal, X), (rul, Y) in product(X_dict.items(), Y_dict.items()):
    #     assert (X.columns==Y.columns).all()

    #     score_dict.setdefault(rul, pd.DataFrame())
    #     score_dict[rul].loc[signal, "Internal Robustness"] = get_internal_robustness(X, Y)
    #     score_dict[rul].loc[signal, "External Robustness"] = get_external_robustness(X, Y)
    #     score_dict[rul].loc[signal, "Reliability"] = get_reliability(X, Y)

    # scaler = MinMaxScaler((0, 100))
    # for rul, score_df in score_dict.items():
    #     # Normalize
    #     score_df.loc[:, :] = scaler.fit_transform(score_df)
    #     # Save
    #     radar_chart(score_df, path)

    # return score_dict


def evaluate_mece(score_df):
    # Scale
    boundary = (0, 100)
    scaler = MinMaxScaler(boundary)
    score_df.loc[:, :] = scaler.fit_transform(score_df)
    score_df.loc[:, score_df.nunique()==1] = boundary[1] # asign the upper bound to those columns with all identical score

    # Get one versus all pair
    pair1, pair2 = [], []
    for hi1, hi2 in product(score_df.index, score_df.index):
        if not score_df.loc[hi1].dropna().size or not score_df.loc[hi2].dropna().size:
            continue
        pair1.append(hi1)
        pair2.append(hi2)
        
    # One versus all distance
    distance = mahalanobis_distance(score_df.loc[pair1].values, score_df.loc[pair2].values)
    mece_df = pd.DataFrame([pair1, pair2, distance]).T.pivot(index=0, columns=1, values=2).astype(float)
    mece_df.index.name = None
    mece_df.columns.name = None

    return mece_df

    """
    # Correlation Analysis
    correlation_matrix = score_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.show()

    # Principal Component Analysis (PCA)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    pca.fit(score_df)
    print("PCA Explained Variance Ratio:", pca.explained_variance_ratio_)
    """
