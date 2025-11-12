

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
# import plotly
# import plotly.tools as tls
# import warnings
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from data_helpers import swap_dict_level


def plot_segment(ser:pd.Series, state_window_dict:pd.Series, dir:str, experiment:str, hi:str):
    title = f'{experiment}, {hi}'
    fig, ax = plt.subplots(figsize=(min(max(ser.size/1000, 9), 16), 4))
    value_count = pd.Series()
    for state, window in state_window_dict.items():
        value_count.loc[state] = len(window)
        plt.scatter(window, ser.loc[window], s=0.5, alpha=0.5)
    plt.title(f'{title} Segmentation')
    plt.xlabel('Time')
    plt.ylabel(hi)
    plt.savefig(dir+f'{title} Segmentation.png', transparent=True, 
                bbox_inches='tight', dpi=288)
    plt.clf()
    plt.close('all')

    value_count.plot(kind='bar')
    plt.title(f'{title} Segment Distribution')
    plt.xlabel('Segment')
    plt.ylabel('Frequency')
    plt.savefig(dir+f'{title} Segment Distribution.png', transparent=True, 
            bbox_inches='tight', dpi=288)
    plt.clf()
    plt.close('all')


def plot_simualtion(health_indicator:pd.Series, simulated_paths:np.array, n_paths_shown:int, dir:str, experiment:str, hi:str):
    title = f'{experiment}, {hi}'
    fig, ax = plt.subplots(figsize=(min(max(health_indicator.size/1000, 9), 16), 4))

    # Constructed health indicator
    plt.plot(health_indicator, color='r', alpha=1, linewidth=1.5, 
             label='Constructed Health Indicator', zorder=1)

    # Simulated paths (label only the first line)
    mse = [mean_squared_error(health_indicator, sim) for sim in simulated_paths]
    to_show = np.argsort(mse)[:n_paths_shown]
    plt.plot(simulated_paths[to_show[0]], linestyle='-.', color='slategrey', 
             alpha=.7, linewidth=.8, label='Simulated Health Indicator', zorder=2)
    for i in to_show:
        plt.plot(simulated_paths[i], linestyle='-.', color='slategrey', 
                 alpha=.7, linewidth=.8, zorder=2)
    
    # Confidence interval
    interval = np.percentile(simulated_paths, (10, 90), axis=0)
    plt.fill_between(
        range(simulated_paths.shape[1]), 
        interval[0], 
        interval[1], 
        color='lightgrey', alpha=.7, linewidth=1, label='Confidence Interval', zorder=0
        )
    # avg = simulated_paths.mean(axis=0).ravel()
    # std = simulated_paths.std(axis=0).ravel()
    # plt.fill_between(
    #     range(simulated_paths.shape[1]), 
    #     avg+2*std, 
    #     avg-2*std, 
    #     color='lightgrey', alpha=.7, linewidth=1, label='Confidence Interval', zorder=0
    #     )

    # interval = np.percentile(simulated_paths, (25, 75), axis=0)
    # plt.fill_between(
    #     range(simulated_paths.shape[1]), 
    #     interval[0], 
    #     interval[1], 
    #     color='lightgrey', alpha=.7, linewidth=1, label='Confidence Interval', zorder=0
    #     )
    
    # Save
    plt.title(f'{title} Simulation')
    plt.xlabel('Time')
    plt.ylabel(hi)
    plt.legend()
    plt.savefig(dir+f'{title} Simulation.png', transparent=True, 
            bbox_inches='tight', dpi=288)
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", category=UserWarning)
    #     plotly.offline.plot(tls.mpl_to_plotly(fig), 
    #                         filename=dir+'simulation.html', 
    #                         auto_open=False)
    plt.clf()
    plt.close('all')


# def plot_domain_health(df, dir, title):
#     for d, health_indicator in hi_dict.items():break
#         plot_health_indicator

#     fig, ax = plt.subplots(figsize=(min(max(df.shape[0]/1000, 9), 16), 4))
#     df.plot(color='r', alpha=.5, lw=1, ax=ax)
#     plt.title(title)
#     plt.savefig(dir+title+'.png', transparent=True, 
#                 bbox_inches='tight', dpi=288)
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", category=UserWarning)
#         plotly.offline.plot(tls.mpl_to_plotly(fig), 
#                             filename=dir+title+'.html', 
#                             auto_open=False)
#     plt.clf()
#     plt.close('all')


def plot_health_indicator(df:pd.DataFrame, dir:str, experiment:str, hi:str):
    title = f'{experiment}, {hi}'
    fig, ax = plt.subplots(figsize=(min(max(df.shape[0]/1000, 9), 16), 4))
    df.plot(color='r', alpha=.8, lw=1, ax=ax)
    plt.title(experiment)
    plt.xlabel('Time')
    plt.ylabel(hi)
    plt.savefig(dir+title+'.png', transparent=True, 
                bbox_inches='tight', dpi=288)
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", category=UserWarning)
    #     plotly.offline.plot(tls.mpl_to_plotly(fig), 
    #                         filename=dir+title+'.html', 
    #                         auto_open=False)
    plt.clf()
    plt.close('all')


def plot_reconstruction(X, X_recon, dir, suffix):
    for signal, x in X.items():
        x_recon = X_recon[signal]
        title = f'{suffix}, {signal}'
        format_text = lambda t: f'{t:.3f}' if t<10**3 and t>=10**-3 else f'{t:.3e}'

        text = f'Original Avg.: {format_text(x.mean())}\
            \nOriginal Std.: {format_text(x.std())}\
            \nReconstructed Avg.: {format_text(x_recon.mean())}\
            \nReconstructed Std.: {format_text(x_recon.std())}\
            \nMSE: {format_text(mse(x, x_recon))}'
        fig, ax = plt.subplots(figsize=(min(max(x.size/1000, 9), 16), 4))
        ax2 = ax.twinx()

        # Reconstruction error
        error = x_recon.values-x.values
        colors = np.full(error.shape, "white")
        colors[error<0] = "r"
        colors[error>=0] = "g"
        ax2.vlines(x=x.index, ymin=0, ymax=np.abs(error), 
                   colors=colors, alpha=0.5, linewidth=0.1, zorder=1)
        ax2.grid(False)
        ax2.set_ylabel('Error')

        # Original and reconstructed points
        ax.plot(x.values, 'b--', alpha=0.8, lw=1,  label='Original Signal')
        ax.plot(x_recon.values, color='grey', alpha=1, lw=1.5, label='Reconstructed Signal')
        ax.set_xlabel('Time')
        ax.set_ylabel(signal)
        ax.legend()

        # ax.text(x_recon.size, x_recon.min(), text)

        plt.title(f'{suffix} Reconstruction')
        plt.savefig(dir+f'{title} Reconstruction.png', transparent=True, 
                    bbox_inches='tight', dpi=288)
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=UserWarning)
        #     plotly.offline.plot(tls.mpl_to_plotly(fig), 
        #                         filename=dir+title+'.html', 
        #                         auto_open=False)
        plt.clf()
        plt.close('all')
            
        # # Reconstruction error
        # title = f'{suffix}, {signal}, Error'
        # error = x_recon.values-x.values
        # text = f'Avg.: {format_text(error.mean())}\
        #     \nStd.: {format_text(error.std())}'
        # fig, ax = plt.subplots(figsize=(min(max(x.size/1000, 9), 16), 4))
        # plt.plot(error, color='r', alpha=0.8, lw=1,  label='Reconstruction Error')
        # plt.legend()
        # plt.text(error.size, error.min(), text)
        # plt.title(title)
        # plt.xlabel('Time')
        # plt.savefig(dir+title+'.png', transparent=True, 
        #             bbox_inches='tight', dpi=288)
        # # with warnings.catch_warnings():
        # #     warnings.filterwarnings("ignore", category=UserWarning)
        # #     plotly.offline.plot(tls.mpl_to_plotly(fig), 
        # #                         filename=dir+title+'.html', 
        # #                         auto_open=False)
        # plt.clf()
        # plt.close('all')


def plot_data(X_dict, dir):
    # One plot for each
    for experiment, X in X_dict.items():
        for signal, x in X.items():
            title = f'{experiment}, {signal}'
            fig, ax = plt.subplots(figsize=(min(max(x.size/1000, 9), 16), 4))
            plt.plot(x, linewidth=1, alpha=1)
            plt.title(experiment)
            plt.xlabel('Time')
            plt.ylabel(signal)
            plt.savefig(dir+title+'.png', transparent=True, 
                        bbox_inches='tight', dpi=288)
            plt.clf()
            plt.close('all')

    # All experiment in one plot
    dict_ = swap_dict_level(X_dict)
    for signal, df in dict_.items():
        title = signal
        plt.plot(df, linewidth=0.5, alpha=0.8)
        plt.legend(df.columns)
        # plt.title(signal)
        plt.xlabel('Time')
        plt.ylabel(signal)
        plt.savefig(dir+title+'.png', transparent=True, 
                    bbox_inches='tight', dpi=288)
        plt.clf()
        plt.close('all')


def plot_radar_chart(score_df, dir):
    # Scale
    boundary = (0, 100)
    scaler = MinMaxScaler(boundary)
    score_df.loc[:, :] = scaler.fit_transform(score_df)
    score_df.loc[:, score_df.nunique()==1] += boundary[1]-score_df.loc[:, score_df.nunique()==1].min() # asign the upper bound to those columns with all identical score
    score_df = score_df.fillna(boundary[0])

    # Plot
    labels = score_df.columns
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    # angles = np.concatenate((angles,[angles[0]]))

    distribution = list(score_df.dropna().T.values)

    for indicator, scores in score_df.iterrows():
        title = indicator if type(indicator) is str else ', '.join(indicator)
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111, polar=True)

        ax.violinplot(
            distribution, angles, points=1000, widths=0.5, 
            showmeans=False, showmedians=False, showextrema=False, 
            )
        plt.setp(ax.collections, alpha=.2)

        ax.plot(angles, scores, 'bo', linewidth=1)
        ax.fill(angles, scores, 'b', alpha=.8)
        ax.set_thetagrids(angles * 180/np.pi, labels)
        
        ax.set_title(f'{title} (AVG.{round(scores.mean(), 2)})')
        ax.grid(True, c='dimgrey')
        plt.xticks(color='dimgrey', size=10)
        plt.yticks(color='grey', size=7)
        plt.savefig(dir+title+'.png', transparent=True, 
                    bbox_inches='tight', dpi=288)
        plt.clf()
        plt.close()


def plot_dissimilarity(mece_df, dir):
    # Plot
    sns.heatmap(mece_df, cmap='rocket_r', annot=True, square=True, fmt='.2g', annot_kws={'size': 6})
    plt.title(f'Dissimilarity')
    plt.savefig(dir+'Dissimilarity.png', transparent=True, 
            bbox_inches='tight', dpi=288)
    plt.clf()
    plt.close('all')

