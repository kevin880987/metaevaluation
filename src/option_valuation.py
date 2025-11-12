

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc


def real_options_analysis(
        simulated_paths:np.array, 
        point:tuple, threshold:tuple, 
        option_valuation:dict, dir:str, title:str):
    alarm_point, failure_point = point
    alarm_threshold, failure_threshold = threshold

    # Iterate every candidate maintenance time interval
    candidate_intervals = option_valuation["I"] * option_valuation["delta_i"]
    candidate_intervals -= candidate_intervals.max() - failure_point
    option_value = {}
    for i, t in enumerate(candidate_intervals):
        if t<0:
            continue

        # Time period to maintain
        extension = option_valuation["delta_i"]*i
        # t = alarm_point + extension - 1

        # Check time length
        if simulated_paths.shape[1]<=t:
            break
        # assert simulated_paths.shape[1]>t, f"Candidate maintenance time t={t} exceeds total time {simulated_paths.shape[1]}."
        
        option_value[t] = {}

        # Number of failures occured before maintenance
        n_failures = np.any(simulated_paths[:, :t+1]>failure_threshold, axis=1).sum()

        # Probability of failure before maintenance
        p = n_failures/simulated_paths.shape[0]
        # if n_failures==0:
        #     p = option_valuation["I"].size*i/simulated_paths.shape[0]
        # else:
        #     p = n_failures/simulated_paths.shape[0]

        # Maintenance lag after actual failure
        maintenance_lag = []
        for path in simulated_paths[simulated_paths[:, t]>failure_threshold, alarm_point:t+1]:
            failure = np.argwhere(path>failure_threshold)
            if failure.size>0:
                maintenance_lag.append(extension-failure[0, 0])
            else:
                maintenance_lag.append(0)
        mean_maintenance_lag = np.nanmean(maintenance_lag) if maintenance_lag else 0

        # Revenues
        R_pr = (1-p)*option_valuation["R_prix"]*extension
        R_cr = p*option_valuation["R_prix"]*(extension-mean_maintenance_lag)
        
        # Costs
        C_pr_sup = (1-p)*option_valuation["C_dm"]*i
        C_cr_sup = p*(option_valuation["C_rep"]+option_valuation["I_penalty"]*mean_maintenance_lag)

        # Calculate
        option_value[t]["Expected Revenue"] = R_pr + R_cr
        option_value[t]["Expected Cost"] = C_pr_sup + C_cr_sup
        option_value[t]["Total Profit"] = option_value[t]["Expected Revenue"]-option_value[t]["Expected Cost"]

    option_values = pd.DataFrame(option_value).T

    # Plot
    plt.figure(figsize=(min(max(option_values.shape[0]/1000, 9), 16), 4))
    option_values.plot(style=["--", "-.", "-"])
    x, y = option_values["Total Profit"].idxmax(), option_values["Total Profit"].max()
    plt.plot(x, y, "ro")
    plt.text(x-option_values.size/20, y+option_values.max().max()/20, f"Optimal: {str(x)}", ha="right", va="bottom")
    plt.ylabel("Amount")
    plt.xlabel("Time")
    plt.title(f"{title}, Option Values")
    plt.savefig(dir+"Option Values.svg", transparent=True, 
                bbox_inches="tight", dpi=288)
    plt.clf()
    plt.close("all")

    # Save
    option_values.to_csv(dir+"option_values.csv", index=True)
    return option_values


def value_options(config, hi_dict: dict):
    """
    Perform real options analysis for maintenance decision optimization.
    
    This function loads simulated paths from deterioration modeling results
    and performs options valuation analysis for each experiment and health indicator.
    
    Args:
        config: Configuration object containing necessary parameters
        hi_dict: Dictionary of health indicators {experiment: {health_indicator: series}}
    """
    print("=" * 50)
    print("REAL OPTIONS VALUATION")
    print("=" * 50)
    print(f"Experiments: {list(hi_dict.keys())}")
    print(f"Health Indicators: {list(next(iter(hi_dict.values())).keys())}")
    print("-" * 50)
    
    # skip=True
    for exp_idx, (e, df) in enumerate(hi_dict.items(), 1):
        print(f"Processing experiment [{exp_idx:2d}/{len(hi_dict):2d}]: {e}")
        
        N = df.shape[0]  # Number of time steps
        alarm_point = config.alarm_point
        if alarm_point < 0:
            alarm_point = max(0, alarm_point + N)
        failure_point = config.failure_point
        if failure_point < 0:
            failure_point = max(0, failure_point + N)

        point = (alarm_point, failure_point)
        descrete_window_size = config.descrete_window_size

        for hi_idx, (hi, health_indicator) in enumerate(df.items(), 1):
            print(f"  Processing HI [{hi_idx:2d}/{len(df.columns):2d}]: {e}, {hi}", end="")
            # if hi=="CVAE_REC_LS_HT2":
            #     skip=False
            # if skip:
            #     continue

            title = f"{e}, {hi}"
            dir = config.deterioration_modeling_dir + f"{title}" + os.sep
            
            # Load simulated paths
            try:
                simulated_paths_df = pd.read_csv(dir+"simulated_paths.csv", index_col=0)
                simulated_paths = simulated_paths_df.values
            except Exception as ex:
                print(f" - SKIPPED (error: {ex})")
                continue

            # Calculate thresholds
            alarm_threshold = health_indicator.iloc[
                max(0, alarm_point - descrete_window_size):alarm_point + 1
            ].max()
            failure_threshold = health_indicator.iloc[
                max(0, failure_point - descrete_window_size):failure_point + 1
            ].mean()
            threshold = (alarm_threshold, failure_threshold)

            # Perform real options analysis
            option_values = real_options_analysis(
                simulated_paths, point, threshold, 
                config.option_valuation, dir, title
            )
            optimal_time = option_values["Total Profit"].idxmax()
            optimal_profit = option_values["Total Profit"].max()
            print(f" - SUCCESS (Optimal time: {optimal_time}, Profit: {optimal_profit:.2f})")

            # Free memory
            del simulated_paths
            gc.collect()

    print("-" * 50)
    print("REAL OPTIONS VALUATION COMPLETED")
    print("=" * 50)
