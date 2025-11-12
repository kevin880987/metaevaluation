

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns
from river.drift import KSWIN
import gc
# from discreteMarkovChain import markovChain
# https://github.com/jkirkby3/pymle
from pymle.models import GeometricBM
# from pymle.sim.Simulator1D import Simulator1D
from pymle.core.TransitionDensity import ExactDensity, KesslerDensity
from pymle.fit.AnalyticalMLE import AnalyticalMLE
from sklearn.preprocessing import MinMaxScaler#, StandardScaler

from visualization import plot_segment, plot_simualtion
from option_valuation import real_options_analysis


def generate_geometric_brownian_motion(hi_ser:np.array, S0:float, mu:np.array, sigma:np.array, dt:np.array, epsilon:float, N:int, n_paths:int):
    """
    Simulates geometric Brownian motion using Euler"s method.
    
    Args:
        S0 (1D array or float): Initial stock price for every path, array of float with the size of n_path or a float.
        mu (float): Drift (expected return) of the stock.
        sigma (float): Volatility (standard deviation) of the stock.
        # T (int): Time period for simulation.
        N (int): Number of time steps
        dt (float): Time step for simulation.
        epislon (float): Probability to overwrite and keep zero drift and volatility.
        n_paths (int): Number of simulation paths to generate.
    
    Returns:
        numpy.ndarray: Array of simulated stock prices.
    """
    if epsilon>0:
        assert hi_ser.size==N, f"Health indicator size mismatch. Expecting {N}, gets {hi_ser.size}."
        assert epsilon<=1, f"Invalid value of epsilon={epsilon}."
    # N = int(T / dt) # Number of time steps
    # dt = T/N
    mu, sigma, dt = np.array(mu).ravel(), np.array(sigma).ravel(), np.array(dt).ravel()
    # https://stats.stackexchange.com/questions/361234/which-formula-for-gbm-is-correct
    dW = np.sqrt(dt) * np.random.randn(n_paths, N) # Wiener process increment

    # Initialize stock price array with initial value
    S = np.zeros((n_paths, N))
    S[:, 0] = S0
    
    # https://medium.com/@polanitzer/path-independent-exponential-brownian-motion-random-walk-process-in-python-simulate-the-future-bb94dbd13a9b    
    for i in range(1, N):
        # Update stock price using Euler"s method
        # # https://hautahi.com/sde_simulation
        # drift = mu[i] * np.abs(S[:, i - 1]) * dt[i]
        # drawing = sigma[i] * np.abs(S[:, i - 1]) * dW[:, i - 1]
        # dS = drift + drawing
        # S[:, i] = S[:, i-1] + dS

        # Update stock price using Milstein"s method
        # https://hautahi.com/sde_simulation
        drift = mu[i] * np.abs(S[:, i-1]) * dt[i]
        drawing = sigma[i] * np.abs(S[:, i-1]) * dW[:, i-1]
        # Flip direction if mu is negative
        correction = np.sign(mu[i]) * 0.5*sigma[i]**2 * np.abs(S[:, i-1]) * (dW[:, i-1] ** 2 - dt[i])
        dS = drift + drawing + correction
        S[:, i] = S[:, i-1] + dS

        if epsilon>0 and not np.isnan(hi_ser[i-1]):
            # Average with the health indicator
            j = np.argwhere(np.random.rand(n_paths)<epsilon).ravel()
            S[j, i] = (hi_ser[i-1]+S[j, i-1])/2 + dS[j]

            # S[:, i] = epsilon*hi_ser[i-1]+(1-epsilon)*S[:, i-1] + dS
        

    # # Shift
    # S[:, :] -= (S[:, 0]-S0)[:, None]
    return S


def estimate_parameter(arr:np.array, dt:float, roll=1):
    arr = np.array(arr).ravel()
    arr = arr[np.isfinite(arr)]

    # https://github.com/jkirkby3/pymle
    # dt = 1/arr.size
    model = GeometricBM()
    # param_bounds = [(-2*abs(arr.mean()), 2*abs(arr.mean())), (0, 2*arr.var())]
    # guess = [arr.mean(), arr.var()]
    param_bounds = [(0, 10), (0, 2)]
    guess = [arr.mean(), arr.std()]

    # Sigma
    kessler_est = AnalyticalMLE(arr-arr.min()+1, param_bounds, dt, density=ExactDensity(model)).estimate_params(guess) # https://github.com/jkirkby3/pymle
    sigma = kessler_est.params[1]

    # Possitive part of mu
    pos = np.array(list(filter(lambda x: x>=0, arr)))
    if pos.size>1:
        kessler_est = AnalyticalMLE(pos, param_bounds, dt, density=ExactDensity(model)).estimate_params(guess) # https://github.com/jkirkby3/pymle
        mu_pos = kessler_est.params[0]
    else:
        mu_pos = 0

    # Negative part of mu
    neg = np.array(list(filter(lambda x: x<0, arr)))
    if neg.size>1:
        neg -= neg.min()-1
        kessler_est = AnalyticalMLE(neg, param_bounds, dt, density=ExactDensity(model)).estimate_params(guess) # https://github.com/jkirkby3/pymle
        mu_neg = -1*kessler_est.params[0]
    else: 
        mu_neg = 0

    # Weighted average of mu
    mu = mu_pos*pos.size/arr.size + mu_neg*neg.size/arr.size

    return mu, sigma


"""
plt.plot(health_indicator)
smoothened.plot()
health_indicator_norm.plot()
increments.plot()
signs.plot()
abs_increments.plot()
log_increments.plot()
log_increments.rolling(descrete_window_size).mean().plot()
variance_of_increments.plot()
mean_of_increments.plot()
plt.plot(np.exp(np.abs(log_increments.loc[window])))
signs.plot()

for state, window in state_window_dict.items():
    print(len(window))
for state, window in state_window_dict.items():
    plt.plot(health_indicator.loc[window], label=state)
plt.legend()
for state, window in state_window_dict.items():
    plt.plot(smoothened.loc[window], label=state)
plt.legend()
for state, window in state_window_dict.items():
    plt.plot(log_increments.loc[window], label=state)
plt.legend()

log_increments[window].plot()
exp_increments.plot()
sign_increments.plot()
exp_series.plot()
plt.plot(mu)
plt.plot(sigma)
plt.plot(dt)
for state, window in state_window_dict.items():
    plt.plot(mu.loc[window], label=state)
plt.legend()
for state, window in state_window_dict.items():
    plt.plot(sigma.loc[window], label=state)
plt.legend()
for state, window in state_window_dict.items():
    plt.plot(dt.loc[window], label=state)
plt.legend()


for s in S:
    plt.plot(s)
for s in simulated_paths:
    plt.plot(s)
"""


def get_transition_matrix(ser:pd.Series, dir:str, title:str):
    # Initilaize
    states = sorted(ser.cat.categories)
    transition_matrix = pd.DataFrame(index=states, columns=states, dtype=float)

    # Get transitions
    next_ser = ser.shift(-1)
    transitions = pd.concat([ser, next_ser], axis=1)

    # Get transition matrix
    counts = transitions.groupby(by=transitions.columns.tolist(), observed=False).value_counts().items()
    for (s, s_next), count in counts:
        transition_matrix.loc[s, s_next] = count/(ser==s).sum()

    # Save
    transition_matrix.to_csv(dir+"transition_matrix.csv")

    # Plot
    sns.heatmap(transition_matrix, annot=True, square=True, fmt=".2g", annot_kws={"size": 6})
    plt.title(f"{title}, Transition Matrix")
    plt.savefig(dir+"Transition Matrix.svg", transparent=True, 
            bbox_inches="tight", dpi=288)
    plt.clf()
    plt.close("all")

    return transition_matrix


def prepare(health_indicator:pd.Series, descrete_window_size:int, dir:str, title:str):
    assert health_indicator.size>descrete_window_size, f"Health indicator size ({health_indicator.size}) smaller than window size ({descrete_window_size})."

    # Smoothened
    smoothened = health_indicator.copy().rolling(descrete_window_size).mean()
    
    # Shift
    # Leave nan in health_indicator_norm to preserve same shape with health_indicator
    health_indicator_norm_trend = smoothened+max(-smoothened.min(), smoothened.iloc[-descrete_window_size:].mean())# if smoothened.min()<=0 else smoothened
    
    # Log increments
    increments = health_indicator_norm_trend/health_indicator_norm_trend.shift()-1
    signs = np.sign(increments)
    abs_increments = np.abs(increments).replace(0, np.nan).dropna()
    log_increments = np.log(abs_increments) # https://blogs.sas.com/content/iml/2014/07/14/log-transformation-of-pos-neg.html

    # increments = np.abs(smoothened-smoothened.shift()).replace(0, np.nan).dropna()
    # signs = np.sign(smoothened)
    # log_increments = signs*(np.log(np.abs(increments))) # https://blogs.sas.com/content/iml/2014/07/14/log-transformation-of-pos-neg.html

    # increments = np.abs(smoothened/smoothened.rolling(descrete_window_size).mean().shift()).replace(0, np.nan).dropna()
    # signs = np.sign(smoothened)
    # log_increments = signs*(np.log(np.abs(increments))) # https://blogs.sas.com/content/iml/2014/07/14/log-transformation-of-pos-neg.html

    plt.figure(figsize=(min(max(log_increments.size/1000, 9), 16), 4))
    log_increments.plot()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Log Increments")
    plt.savefig(dir+"Log Increments.svg", transparent=True, 
            bbox_inches="tight", dpi=288)
    plt.clf()
    plt.close("all")

    # Variance of increments
    variance_of_increments = log_increments.rolling(descrete_window_size, closed="right").var()
    # variance_of_increments = log_increments.groupby(np.arange(log_increments.size)//descrete_window_size).var()
    plt.figure(figsize=(min(max(variance_of_increments.size/1000, 9), 16), 4))
    variance_of_increments.plot()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Variance of Increments")
    plt.savefig(dir+"Variance of Increments.svg", transparent=True, 
            bbox_inches="tight", dpi=288)
    plt.clf()
    plt.close("all")

    # Mean of increments
    mean_of_increments = log_increments.rolling(descrete_window_size, closed="right").mean()
    # mean_of_increments = log_increments.groupby(np.arange(log_increments.size)//descrete_window_size).mean()
    plt.figure(figsize=(min(max(mean_of_increments.size/1000, 9), 16), 4))
    mean_of_increments.plot()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Mean of Increments")
    plt.savefig(dir+"Mean of Increments.svg", transparent=True, 
            bbox_inches="tight", dpi=288)
    plt.clf()
    plt.close("all")

    return health_indicator_norm_trend, log_increments, signs

def categorize(ser:pd.Series, descrete_window_size:int, dir:str, title:str):
    # Descretized
    # # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    # index = variance_of_increments.index
    # stats = np.concatenate([
    #     # MinMaxScaler().fit_transform(smoothened.loc[index].to_frame()), 
    #     # MinMaxScaler().fit_transform(log_increments.to_frame()), 
    #     MinMaxScaler().fit_transform(variance_of_increments.to_frame()), 
    #     MinMaxScaler().fit_transform(mean_of_increments.to_frame()), 
    #     ], axis=1)
    # stats = pd.DataFrame(stats, index=index).dropna()
    # # descretized = DBSCAN(eps=0.01, min_samples=30).fit_predict(stats)
    # descretized = MeanShift(min_bin_freq=descrete_window_size, max_iter=1500).fit_predict(stats)
    # descretized = pd.Series(descretized, index=stats.index).astype(int).astype("category")
    variance = ser.iloc[-descrete_window_size:].var() # according to the origin paper
    variance_of_increments = ser.rolling(descrete_window_size, closed="right").var()
    categorized = pd.cut(
        variance_of_increments, bins=[0, variance, 2*variance, np.inf], 
        right=True, labels=["low", "normal", "high"], 
        )
    categorized.value_counts().plot(kind="bar")
    plt.title(f"{title}, Categorized Counts")
    plt.savefig(dir+"Categorized Counts.svg", transparent=True, 
            bbox_inches="tight", dpi=288)
    plt.clf()
    plt.close("all")

    # Seperate states
    state_window_dict = {}
    for state, window in categorized.groupby(by=categorized, observed=False):
        state_window_dict[state] = window.index.tolist()
 
    return state_window_dict, categorized

def segment(ser:pd.Series, descrete_window_size:int):
    state_window_dict = {}
    state = 0
    ids = []
    kswin = KSWIN(alpha=0.01, window_size=descrete_window_size*3, stat_size=descrete_window_size, seed=42)
    for i, (id, element) in enumerate(ser.items()):
        kswin.update(element)
        ids.append(id)
        if kswin.drift_detected or i==ser.size-1:
            state_window_dict[state] = ids
            state += 1
            ids = []

    return state_window_dict


def model_deterioration(config, hi_dict:dict):
    n_paths = config.n_paths
    n_paths_shown = config.n_paths_shown
    epsilon = config.epsilon

    print("=" * 50)
    print("DETERIORATION MODELING")
    print("=" * 50)
    print(f"Method: {config.deterioration_method}")
    print(f"Paths to generate: {n_paths}")
    print(f"Epsilon (health indicator mixing): {epsilon}")
    print(f"Experiments: {list(hi_dict.keys())}")
    print("-" * 50)

    for exp_idx, (e, df) in enumerate(hi_dict.items(), 1):
        print(f"Processing experiment [{exp_idx:2d}/{len(hi_dict):2d}]: {e}")
        
        N = df.shape[0]  # Number of time steps
        # alarm_point = config.alarm_point
        # if alarm_point < 0:
        #     alarm_point = max(0, alarm_point + N)
        # failure_point = config.failure_point
        # if failure_point < 0:
        #     failure_point = max(0, failure_point + N)

        # point = (alarm_point, failure_point)
        descrete_window_size = config.descrete_window_size

        for hi_idx, (hi, health_indicator) in enumerate(df.items(), 1):
            print(f"  Processing HI [{hi_idx:2d}/{len(df.columns):2d}]: {e}, {hi}", end="")
            
            title = f"{e}, {hi}"
            dir = config.deterioration_modeling_dir + f"{title}" + os.sep
            config._make_directories([dir])
            
            # # Prepare health indicator
            # health_indicator_norm_trend, log_increments, signs = prepare(health_indicator, descrete_window_size, dir, title)
            
            # # Starting value of simulation
            # S0 = health_indicator_norm_trend.min()

            # # Scale health indicator
            # health_indicator_norm = health_indicator.rolling(descrete_window_size).mean().dropna()
            # hi_scaler = MinMaxScaler((0, 1))
            # hi_scaler.fit(health_indicator_norm.to_frame())            

            # if config.deterioration_method=="EWCD":
            #     state_window_dict = segment(log_increments, descrete_window_size)
                
            #     # Plot segmentation
            #     plot_segment(health_indicator, state_window_dict, dir, e, hi)

            #     # Simulate for each state respectively
            #     mu = pd.Series(index=df.index)
            #     sigma = pd.Series(index=df.index)
            #     dt = pd.Series(index=df.index)
            #     for state, window in state_window_dict.items():
            #         if len(window)<descrete_window_size: continue
                    
            #         # ser = log_increments[window] # log_increments.rolling(config.descrete_window_size).mean()[window] # 
            #         # signs = np.sign(ser)
            #         # exp_series = signs*(np.exp(np.abs(ser))-1).cumprod()

            #         # series = health_indicator_norm.loc[window]
            #         # series = (series/series.shift()).dropna()
            #         # signs = np.sign(series)
            #         # log_increments = np.log(np.abs(series))
            #         # exp_series = signs*(np.exp(np.abs(log_increments)))
                    
            #         # signs = np.sign(health_indicator.loc[window])
            #         # exp_series = signs*(np.exp(np.abs(log_increments.loc[window]))-1)

            #         exp_increments = np.exp(log_increments.loc[window])
            #         sign_increments = signs*exp_increments
            #         exp_series = sign_increments.cumsum()

            #         # Estimate parameters
            #         mu.loc[window], sigma.loc[window] = estimate_parameter(
            #             exp_series, dt=1/N, # 1/len(window), #len(window)/descrete_window_size, #N/len(window) # roll=descrete_window_size, 
            #             )
            #         # mu.loc[window] += np.log(offset/len(window)) # to compensate the offset that ensures positive values in log transformation
            #         dt.loc[window] = 1/N # 1/len(window) # len(window)/descrete_window_size # N/len(window) # 
                                        
            #     mu, sigma, dt = mu.bfill(), sigma.bfill(), dt.bfill()
            #     mu, sigma, dt = mu.ffill(), sigma.ffill(), dt.ffill()
            #     simulated_paths = generate_geometric_brownian_motion(
            #         health_indicator_norm_trend.values, S0, mu, sigma, dt, epsilon, N, n_paths, 
            #         )

            # elif config.deterioration_method=="EWVOI":
            #     state_window_dict, categorized = categorize(log_increments, descrete_window_size, dir, title)
            #     categorized = categorized.dropna().cat.remove_unused_categories()

            #     # Plot segmentation
            #     plot_segment(health_indicator, state_window_dict, dir, e, hi)

            #     # # Calculate transition matrix
            #     # transition_matrix = get_transition_matrix(categorized, dir, title)

            #     # # Calculate Markov chain
            #     # # https://pypi.org/project/discreteMarkovChain/
            #     # mc = markovChain(transition_matrix.values)
            #     # mc.computePi("linear")
            #     # steady_state = mc.pi

            #     weights = categorized.value_counts()/categorized.value_counts().sum()

            #     # Simulate for each state respectively
            #     simulated_paths = []
            #     for state, weight in weights.items():
            #         # if state=="low":break
            #         window = state_window_dict[state]
            #         # print(state, len(window))
            #         if len(window)<descrete_window_size or weight==0: continue

            #         # ser = log_increments[window] # log_increments.rolling(config.descrete_window_size).mean()[window] # 
            #         # signs = np.sign(ser)
            #         # # exp_series = signs*(np.exp(np.abs(ser))-1).cumprod()
            #         # exp_series = signs*(np.exp(np.abs(ser))).cumsum()
            #         # # series = health_indicator_norm.loc[window]
            #         # # series = (series/series.shift()).dropna()
            #         # # signs = np.sign(series)
            #         # # log_increments = np.log(np.abs(series))
            #         # # exp_series = signs*(np.exp(np.abs(log_increments)))

            #         exp_increments = np.exp(log_increments.loc[window])
            #         sign_increments = signs*exp_increments
            #         exp_series = sign_increments.cumsum()

            #         # Estimate parameters
            #         mu, sigma = estimate_parameter(
            #             exp_series, dt=1/N, # 1/len(window), # len(window)/descrete_window_size, # N/len(window) # roll=descrete_window_size, 
            #             )
            #         dt = 1/N # 1/len(window) # N/len(window) # 

            #         # Generate
            #         mu = pd.Series(mu, index=df.index)
            #         sigma = pd.Series(sigma, index=df.index)
            #         dt = pd.Series(dt, index=df.index)
            #         simulated_paths.append(generate_geometric_brownian_motion(
            #             health_indicator_norm_trend.values, S0, mu, sigma, dt, epsilon, N, n_paths, 
            #             )*weight)

            #     # Weighted average
            #     simulated_paths = np.sum(simulated_paths, axis=0)

            # # elif config.deterioration_method=="EW":
            # #     # Simulate
            # #     mu, sigma = estimate_parameter(
            # #         health_indicator, dt=1, #roll=descrete_window_size, 
            # #         )
            # #     simulated_paths = generate_geometric_brownian_motion(
            # #         health_indicator, S0, mu, sigma, dt=1, n_paths=n_paths, 
            # #         )
            # #     parameter_df = pd.concat([
            # #         parameter_df, 
            # #         pd.Series(
            # #             [e, hi, config.deterioration_method, "", mu, sigma], 
            # #             index=columns,
            # #             ).to_frame().N, 
            # #             ], ignore_index=True)
                
            # #     # # Scale
            # #     # sim_scaler = MinMaxScaler()
            # #     # simulated_paths = sim_scaler.fit_transform(simulated_paths.T).T
            # #     # simulated_paths = hi_scaler.inverse_transform(simulated_paths.T).T

            # else:
            #     break

            # # Scale
            # sim_norm = np.lib.stride_tricks.sliding_window_view(simulated_paths, descrete_window_size, axis=1)\
            #     .mean(axis=2).mean(axis=0).reshape(-1, 1)
            # sim_scaler = MinMaxScaler()
            # sim_scaler.fit(sim_norm)
            # simulated_paths = np.array([sim_scaler.transform(s.reshape(-1, 1)).ravel() for s in simulated_paths])
            # simulated_paths = hi_scaler.inverse_transform(simulated_paths.T).T

            # # Save
            # simulated_paths_df = pd.DataFrame(simulated_paths)
            # simulated_paths_df.to_csv(dir+"simulated_paths.csv", index=True)
            
            # Load
            simulated_paths_df = pd.read_csv(dir+"simulated_paths.csv", index_col=0)
            simulated_paths = simulated_paths_df.values

            # PLot simulation
            plot_simualtion(health_indicator, 
                            simulated_paths, 
                            n_paths_shown, dir, e, hi)

            # # Get alarm and failure
            # alarm_threshold = health_indicator.iloc[max(0, alarm_point-descrete_window_size):alarm_point+1].max()
            # # failure_point = N-descrete_window_size
            # # failure_threshold = np.nanmean(health_indicator.iloc[failure_point:])
            # failure_threshold = health_indicator.iloc[max(0, failure_point-descrete_window_size):failure_point+1].mean()
            # threshold = (alarm_threshold, failure_threshold)

            # # Real option
            # _ = real_options_analysis(simulated_paths, point, 
            #                       threshold, config.option_valuation, 
            #                       dir, title)

            # Free memory
            del simulated_paths
            gc.collect()

    # Save
    # parameter_df.to_csv(config.deterioration_modeling_dir+f"estimated_parameters.csv")