

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from configuration import Configuration
from data_holder import DataHolder


def plot_rul_distribution(config, holder):
    deterioration_modeling_dir = config.result_dir+"deterioration modeling"+os.sep
    ewcd_dir = deterioration_modeling_dir+"EWCD"+os.sep
    ewvoi_dir = deterioration_modeling_dir+"EWVOI"+os.sep

    # Load data
    dir = config.data_dir
    holder = DataHolder()
    holder.read_files(dir)

    experiments = holder.experiments # ["1"] # 
    indicators = config.health_indicators # ["REC MD"] # ["PC1"] # 

    for e in experiments:
        health_indicators = pd.read_csv(config.health_indicator_dir+e+os.sep+f"{e}.csv", index_col=0)

        for hi in indicators:
            
            health_indicator = health_indicators[hi]
            # alarm_threshold = health_indicator.iloc[config.alarm_point]
            failure_threshold = health_indicator.iloc[config.alarm_point-config.descrete_window_size:config.alarm_point+1].mean()

            ewcd_sim = pd.read_csv(ewcd_dir+f"{e}, {hi}"+os.sep+"simulated_paths.csv", index_col=0)
            candidate = ewcd_sim#.iloc[:, config.alarm_point:]
            ewcd_rul = (candidate>failure_threshold)\
                    .apply(lambda x: x.idxmax() if np.any(x) else np.nan, axis=1)\
                    .dropna().astype(int)

            ewvoi_sim = pd.read_csv(ewvoi_dir+f"{e}, {hi}"+os.sep+"simulated_paths.csv", index_col=0)
            candidate = ewvoi_sim#.iloc[:, config.alarm_point:]
            ewvoi_rul = (candidate>failure_threshold)\
                    .apply(lambda x: x.idxmax() if np.any(x) else np.nan, axis=1)\
                    .dropna().astype(int)

            title = f"{e}, {hi}, RUL Distribution"
            dir = deterioration_modeling_dir+"distribution"+os.sep#+f"{e}, {hi}"+os.sep
            os.makedirs(dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(min(max(100/1000, 9), 16), 4))
            ewcd_rul.plot.hist(
                bins=100, alpha=0.4, 
                label=f"EWCD (avg:{round(ewcd_rul.mean(), 2)} std:{round(ewcd_rul.std(), 2)})", 
                )
            ewvoi_rul.plot.hist(
                bins=100, alpha=0.4, 
                label=f"EWVOI (avg:{round(ewvoi_rul.mean(), 2)} std:{round(ewvoi_rul.std(), 2)})", 
                )
            plt.legend()
            plt.title(title)
            plt.xlabel("RUL")
            plt.ylabel("Frequency")
            plt.savefig(dir+f"{title}.png", transparent=True, 
                    bbox_inches="tight", dpi=288)
            plt.clf()
            plt.close("all")

    # ewcd_sim.std(axis=1).mean()
    # ewvoi_sim.std(axis=1).mean()
    # ewcd_rul.value_counts()

    # ewcd_sim.T.plot()
    # ewvoi_sim.T.plot()

    # ((candidate>failure_threshold)*candidate.columns).idxmin(axis=1).astype(int)
    # ((candidate>failure_threshold)*candidate.columns)["1093"].iloc[0]
    # (candidate>failure_threshold).apply(lambda x: np.where(x==True)[0][0] if np.any(x) else np.nan, axis=1)
    # ewcd_sim["1093"].mean()

