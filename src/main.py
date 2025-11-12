

# import os
# os.chdir(r"G:\我的雲端硬碟\Academic\PhD\Projects\Index Quality\src")

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from configuration import Configuration
from data_holder import DataHolder
from data_helpers import save_pkl, load_pkl
from health_indicator_extraction import extract
from scoring import score, evaluate_mece
from visualization import plot_data, plot_radar_chart, plot_dissimilarity
from deterioration_modeling import model_deterioration
from option_valuation import value_options
from rul_distribution import plot_rul_distribution
from ensemble import compare_ensembles
from compare_profit import compare_profit

sns.set_theme()
plt.rcParams["font.family"] = "serif"


def main():
    config = Configuration()

    # Load data
    dir = config.data_dir
    holder = DataHolder()
    holder.read_files(dir)

    # Prepare data
    test_ratio = config.test_ratio
    # window_size = config.window_size
    input_sequence = config.input_sequence
    output_sequence = config.output_sequence
    holder.prepare(test_ratio, input_sequence, output_sequence)#, window_size

    # # Plot signals
    # dir = config.original_signal_dir
    # X_dict = holder.item["X"]
    # plot_data(X_dict, dir)

    # # Train model and extract health indicator
    # dir = config.health_indicator_dir
    # hi_dict = extract(config, holder, checkpoint=True) # checkpoint=False) # 
    # save_pkl(hi_dict, dir+"hi_dict.pkl")

    # # Score indicator quality
    # dir = config.health_indicator_dir
    # hi_dict = load_pkl(dir+"hi_dict.pkl")
    # rul_dict = {}
    # for e in holder.experiments:
    #     rul_dict[e] = dict(holder.get("Y", experiment=e, output_sequence=config.output_sequence))
    # score_df = score(hi_dict, rul_dict)
    # score_df.to_csv(dir+"score.csv")

    # # Visualize score
    # dir = config.health_indicator_dir
    # score_df = pd.read_csv(dir+"score.csv", index_col=0)
    # plot_radar_chart(score_df, dir)

    # # MECE
    # dir = config.health_indicator_dir
    # score_df = pd.read_csv(dir+"score.csv", index_col=0)
    # mece_df = evaluate_mece(score_df)
    # mece_df.to_csv(dir+"mece.csv")
    # plot_dissimilarity(mece_df, dir)

    # Deterioration modeling
    dir = config.health_indicator_dir
    hi_dict = load_pkl(dir+"hi_dict.pkl")
    model_deterioration(config, hi_dict)
    
    # # Value options analysis
    # dir = config.health_indicator_dir
    # hi_dict = load_pkl(dir+"hi_dict.pkl")
    # value_options(config, hi_dict)
    
    # # # Plot RUL distribution
    # # plot_rul_distribution(config, holder)

    # # Compare ensembles
    # dir = config.health_indicator_dir
    # hi_dict = load_pkl(dir+"hi_dict.pkl")
    # score_df = pd.read_csv(dir+"score.csv", index_col=0)
    # compare_ensembles(config, hi_dict, score_df)

    # # Compare profit
    # dir = config.health_indicator_dir
    # hi_dict = load_pkl(dir+"hi_dict.pkl")
    # compare_profit(config, hi_dict)

if __name__=="__main__":
    # main()
    pass

