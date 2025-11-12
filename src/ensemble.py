

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

from option_valuation import real_options_analysis


@dataclass
class WeightingParams:
    """Weights for meta-evaluation metrics and softmax temperature.
    The five lambda weights should sum to 1.0.
    """
    lambda_M: float = 0.20
    lambda_P: float = 0.10
    lambda_T: float = 0.30
    lambda_R: float = 0.30
    lambda_IR: float = 0.10
    eta: float = 3.0          # softmax temperature (higher => sparser weights)
    # min_monotonicity: float = 0.2  # drop HIs below this M threshold
    # penalize_instability: bool = True  # downweight by 1/CV of RUL samples

# @dataclass
# class DecisionParams:
#     """Decision grid and risk settings for selecting tau*.
#     tau_grid: array of candidate maintenance times (same units as RUL)
#     risk_alpha: CVaR level for reporting (does not affect tau* unless risk_averse=True)
#     risk_averse: if True, choose tau* that maximizes CVaR instead of mean value
#     """
#     tau_grid: Optional[np.ndarray] = None
#     risk_alpha: float = 0.1
#     risk_averse: bool = False

# ------------------------------------------------------------
# Helpers: metric normalization and weights
# ------------------------------------------------------------
METRIC_ALIASES = {
    'Monotonicity': ['Monotonicity', 'monotonicity', 'M'],
    'Prognosability': ['Prognosability', 'prognosability', 'P'],
    'Trendability': ['Trendability', 'trendability', 'T'],
    'Reliability': ['Reliability', 'reliability', 'R'],
    'InternalRobustness': ['Internal Robustness', 'InternalRobustness', 'internal_robustness', 'IR']
}


def _find_col(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in df.columns:
            return k
    return None


def normalize_scores(score_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all meta-metrics are in [0,1]. If already bounded, return as-is.
    If a metric has negative values, min-max normalize based on observed range.
    """
    df = score_df.copy()
    cols = {}
    for canon, aliases in METRIC_ALIASES.items():
        col = _find_col(df, aliases)
        if col is None:
            raise ValueError(f"Missing metric column for {canon} (aliases: {aliases})")
        cols[canon] = col

    for canon, col in cols.items():
        v = df[col].astype(float).to_numpy()
        # if clearly bounded [0,1], keep; else min-max
        if np.nanmin(v) < 0.0 or np.nanmax(v) > 1.0:
            lo, hi = np.nanmin(v), np.nanmax(v)
            if hi - lo < 1e-12:
                df[col] = 0.0
            else:
                df[col] = (v - lo) / (hi - lo)
        else:
            df[col] = v
    return df


def compute_hi_weights(score_df: pd.DataFrame,
                        # rul_samples: Dict[str, np.ndarray],
                        params: WeightingParams) -> pd.Series:
    """Compute softmax weights for each HI using meta-metrics + (optional) stability penalty.
    score_df: index (HI names) must align with rul_samples keys; columns contain meta-metrics.
    rul_samples: mapping HI name -> 1D array of RUL Monte Carlo samples (>= a few hundred is better).
    Returns: pd.Series of weights summing to 1.
    """
    df = normalize_scores(score_df)

    # Retrieve columns (after normalization these contain 0..1)
    cM = _find_col(df, METRIC_ALIASES['Monotonicity'])
    cP = _find_col(df, METRIC_ALIASES['Prognosability'])
    cT = _find_col(df, METRIC_ALIASES['Trendability'])
    cR = _find_col(df, METRIC_ALIASES['Reliability'])
    cIR = _find_col(df, METRIC_ALIASES['InternalRobustness'])

    lam = np.array([params.lambda_M, params.lambda_P, params.lambda_T, params.lambda_R, params.lambda_IR], dtype=float)
    lam = lam / lam.sum()

    # Compose score S_j = sum lambda_i * metric_i
    S = lam[0]*df[cM] + lam[1]*df[cP] + lam[2]*df[cT] + lam[3]*df[cR] + lam[4]*df[cIR]

    # # Filter: drop HIs with very low monotonicity
    # keep = df[cM] >= params.min_monotonicity
    # if not keep.any():
    #     # If all fail, keep all but warn by normalizing monotonicity threshold away
    #     keep = pd.Series(True, index=df.index)
    # S = S.where(keep, other=np.nan)

    # # Optional stability penalty via 1/CV of RUL distribution
    # if params.penalize_instability:
    #     adj = []
    #     for name in df.index:
    #         x = np.asarray(rul_samples.get(name, np.array([])), dtype=float)
    #         if x.size < 5 or not np.isfinite(x).all():
    #             adj.append(1.0)
    #         else:
    #             mu = float(np.mean(x))
    #             sd = float(np.std(x, ddof=1))
    #             cv = sd / max(abs(mu), 1e-12)
    #             adj.append(1.0 / max(cv, 1e-3))  # higher stability => larger factor
    #     adj = pd.Series(adj, index=df.index)
    #     # scale adj to [0.5, 1.5] to avoid extremes
    #     a = adj.replace([np.inf, -np.inf], np.nan).fillna(adj[adj.replace([np.inf,-np.inf], np.nan).notna()].median())
    #     lo, hi = a.min(), a.max()
    #     if hi - lo < 1e-12:
    #         a = pd.Series(1.0, index=a.index)
    #     else:
    #         a = 0.5 + (a - lo) / (hi - lo)  # 0.5..1.5
    #     S = S * a

    # Softmax over available S (ignore NaNs)
    S_np = S.to_numpy(dtype=float)
    mask = np.isfinite(S_np)
    if mask.sum() == 0:
        # fallback uniform
        w = np.ones_like(S_np) / len(S_np)
    else:
        S_f = S_np[mask]
        z = (S_f - np.nanmax(S_f)) * params.eta
        e = np.exp(z)
        w_part = e / e.sum()
        w = np.zeros_like(S_np)
        w[mask] = w_part
        # If some were filtered, re-normalize across all (zeros for dropped)
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones_like(S_np) / len(S_np)

    return pd.Series(w, index=df.index, name='weight')


def ensemble_health_indicators(config, e, weight, health_indicators, use_n_paths=None, suffix=""):
    if use_n_paths is None:
        use_n_paths = config.n_paths
    else:
        use_n_paths = min(use_n_paths, config.n_paths)

    # Directories
    deterioration_modeling_dir = config.result_dir+"deterioration modeling"+os.sep
    ewcd_dir = deterioration_modeling_dir+"EWCD"+os.sep
    ewvoi_dir = deterioration_modeling_dir+"EWVOI"+os.sep
    dir = deterioration_modeling_dir+"ensemble"+os.sep+f"{e}, {suffix}"+os.sep
    os.makedirs(dir, exist_ok=True)

    ens_simulated_paths = None
    ens_alarm_threshold, ens_failure_threshold = None, None
    for hi, health_indicator in health_indicators.items():
        title = f"{e}, {hi}"

        # Load
        try:
            # EWCD
            ewcd_simulated_paths_df = pd.read_csv(ewcd_dir+f"{title}"+os.sep+"simulated_paths.csv", 
                                                index_col=0, skiprows=config.n_paths-use_n_paths)
            ewcd_simulated_paths = ewcd_simulated_paths_df.values
        except:
            ewcd_simulated_paths = np.tile(health_indicator.values.ravel(), (use_n_paths, 1))    # EWVOI
        try:
            ewvoi_simulated_paths_df = pd.read_csv(ewvoi_dir+f"{title}"+os.sep+"simulated_paths.csv", 
                                                index_col=0, skiprows=config.n_paths-use_n_paths)
            ewvoi_simulated_paths = ewvoi_simulated_paths_df.values
        except:
            ewvoi_simulated_paths = np.tile(health_indicator.values.ravel(), (use_n_paths, 1))

        # Get mu and std from health indicator
        mu = health_indicator.mean()
        std = health_indicator.std()
            
        # Standardize simulated paths using health indicator statistics
        health_indicator = (health_indicator - mu) / std
        ewcd_simulated_paths = (ewcd_simulated_paths - mu) / std
        ewvoi_simulated_paths = (ewvoi_simulated_paths - mu) / std

        # Weighted average
        if ens_simulated_paths is None:
            ens_simulated_paths = weight[hi] * (0.5 * ewcd_simulated_paths + 0.5 * ewvoi_simulated_paths)
        else:
            ens_simulated_paths += weight[hi] * (0.5 * ewcd_simulated_paths + 0.5 * ewvoi_simulated_paths)

        # Locate alarm and failure
        N = ens_simulated_paths.shape[1] # (int): number of time step
        alarm_point = config.alarm_point
        if alarm_point<0:
            alarm_point = max(0, alarm_point+N)
        failure_point = config.failure_point
        if failure_point<0:
            failure_point = max(0, failure_point+N)

        point = (alarm_point, failure_point)
        descrete_window_size = config.descrete_window_size

        # Get alarm and failure
        alarm_threshold = health_indicator.iloc[max(0, alarm_point-descrete_window_size):alarm_point+1].max()
        # failure_point = N-descrete_window_size
        # failure_threshold = np.nanmean(health_indicator.iloc[failure_point:])
        failure_threshold = health_indicator.iloc[max(0, failure_point-descrete_window_size):failure_point+1].mean()

        # Weighted average
        if ens_alarm_threshold is None and ens_failure_threshold is None:
            ens_alarm_threshold = weight[hi] * alarm_threshold
            ens_failure_threshold = weight[hi] * failure_threshold
        else:
            ens_alarm_threshold += weight[hi] * alarm_threshold
            ens_failure_threshold += weight[hi] * failure_threshold

    threshold = (ens_alarm_threshold, ens_failure_threshold)

    # Real option
    title = f"{e}, {suffix}" if suffix else e
    option_values = real_options_analysis(ens_simulated_paths, point, 
                          threshold, config.option_valuation, 
                          dir, title)
    return option_values


def compare_ensembles(config, hi_dict, score_df):
    for e, health_indicators in hi_dict.items():
        wparams = WeightingParams()
        
        suffix = "Metaevaluation"
        if "ieee-phm-2012" in config.dataset:
            indicators = [ # ieee 2012
                "PC1",

                "IC_MD",
                "IC_HT2",

                "CVAE_REC_ERR",
                "CVAE_REC_MD",
                "CVAE_REC_HT2",

                "CVAE_LS_MD",
                "CVAE_LS_HT2",

                "CVAE_REC_LS_MD",
                "CVAE_REC_LS_HT2",
                ]
        elif "NTUST" in config.dataset:
            indicators = [ # NTUST
                "PC1",

                "IC_HT2",

                "CVAE_REC_ERR",
            ]
        weight = compute_hi_weights(score_df.loc[indicators], wparams)
        option_values = ensemble_health_indicators(
            config, e, weight, 
            health_indicators[indicators], 
            use_n_paths=1000, suffix=suffix, 
            )

        suffix = "CAEM"
        indicators = [
            "CAEM_ANOM"
            ]
        weight = compute_hi_weights(score_df.loc[indicators], wparams)
        option_values = ensemble_health_indicators(
            config, e, weight, 
            health_indicators[indicators], suffix=suffix, 
            )
        
        suffix = "VQVAE"
        indicators = [
            "VQVAE_QUANT_ERR"
            ]
        weight = compute_hi_weights(score_df.loc[indicators], wparams)
        option_values = ensemble_health_indicators(
            config, e, weight, 
            health_indicators[indicators], suffix=suffix, 
            )
