

import os
import pandas as pd


def compare_profit(config, hi_dict):
    methods = ['Metaevaluation', 'CAEM', 'VQVAE']
    columns = ['Optimal Maintenance Time (10s)', 'Total Profit']
    deterioration_modeling_dir = config.result_dir+"deterioration modeling"+os.sep

    df = pd.DataFrame(index=hi_dict.keys(), columns=pd.MultiIndex.from_product([methods, columns]))
    for e, health_indicators in hi_dict.items():

        def process_experiment(method):
            dir = deterioration_modeling_dir+"ensemble"+os.sep+f"{e}, {method}"+os.sep
            exp_df = pd.read_csv(os.path.join(dir, 'option_values.csv'), index_col=0)
            exp_sum = pd.Series([
                exp_df.index[exp_df['Total Profit'].argmax()],
                exp_df['Total Profit'].max(),
            ], index=columns, name=e).to_frame().T
            exp_sum.columns = pd.MultiIndex.from_product([[method], columns])
            return exp_sum

        for method in methods:
            exp_sum = process_experiment(method)
            df.loc[e, exp_sum.columns] = exp_sum.values

        actual_life = health_indicators.index[-1]
        df.loc[e, ('Actual Life (10s)', '')] = actual_life

    df.sort_index().to_csv(deterioration_modeling_dir+"summary.csv")