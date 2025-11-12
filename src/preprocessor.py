

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler #, RobustScaler
from kneed import KneeLocator
# from xgboost import XGBRegressor
# from aenet import AdaptiveElasticNet
# from itertools import product
# import warnings
from matplotlib import pyplot as plt
import seaborn as sns

from visualization import plot_data
from scoring import get_monotonicity, get_prognosability, get_trendability, get_reliability, get_external_robustness, get_internal_robustness


# def adaptive_elasticnet(X, Y):
#     """
#     The function fits an adaptive elastic net model to the
#     input data and returns a sorted series of feature importances.
    
#     Args:
#       X (pandas.DataFrame): The input features as a pandas DataFrame.
#       Y (pandas.DataFrame): The target variable or dependent variable in the dataset.
    
#     Returns:
#       pandas.Series: A Pandas Series object containing the feature importance values for each
#     feature in the input data X, as determined by the Adaptive Elastic Net
#     algorithm. The feature importance values are sorted in descending order.
#     """
#     aenet = AdaptiveElasticNet().fit(X, Y.squeeze())

#     feature_importance = pd.Series(aenet.coef_, name="aenet", index=X.columns)
#     feature_importance = feature_importance.sort_values(ascending=False) # sort importance in descending order
    
#     return feature_importance


# def extreme_gradient_boosting(X, Y):
#     """
#     The function performs extreme gradient boosting feature selection on input data
#     and returns a sorted list of feature importances.
    
#     Args:
#       X (pandas.DataFrame): The input features as a pandas DataFrame.
#       Y (pandas.DataFrame): The target variable or dependent variable in the dataset.
    
#     Returns:
#       pandas.Series: A pandas Series object containing the feature importances calculated using
#     extreme gradient boosting algorithm.
#     """
#     # with warnings.filterwarnings("ignore", category=FutureWarning):
#     xgb = XGBRegressor(
#         random_state=42, 
#         nthread=6, 
#         )
#     xgb = xgb.fit(X, Y)

#     feature_importance = pd.Series(
#         xgb.get_booster().get_score(importance_type="gain"), 
#         name="xgb", 
#         )
#     feature_importance = feature_importance.sort_values(ascending=False) # sort importance in descending order
    
#     return feature_importance


def select_by_knee(feature_importance, 
                   params={"curve": "convex", "direction": "decreasing"}, 
                   min_features=1, output_directory="", suffix="", visualize=True):
    if feature_importance.shape[0]<=1:
        return list(feature_importance.index)

    feature_importance = feature_importance.sort_values(ascending=False)
    kn = KneeLocator(
        x=range(feature_importance.shape[0]), 
        y=feature_importance.values.ravel(), 
        **params, 
        )
    if kn.knee:
        knee = min(max(kn.knee, min_features-1), feature_importance.shape[0]-1)
    else:
        knee = feature_importance.shape[0]-1
    selected_features = feature_importance.index[: knee+1].tolist() # *5

    feature_selection = pd.DataFrame(index=feature_importance.index, columns=[suffix])
    feature_selection.loc[selected_features, suffix] = True
    feature_selection = feature_selection.fillna(False)

    if visualize:
        sns.set()
        plt.rcParams["font.family"] = "serif"

        n_features = 50
        title = f"Feature Iimportance {suffix}"
        fig, ax = plt.subplots(figsize=(12, n_features/5))
        sns.barplot(
            x=feature_importance[: n_features].values, 
            y=feature_importance[: n_features].index,
            # capsize=.4, errcolor=".5",
            linewidth=3, edgecolor=".5", facecolor=(0, 0, 0, 0), 
            ax=ax, 
        )        
        plt.title(title)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.savefig(output_directory+title+".png", transparent=True, 
                    bbox_inches="tight", dpi=144)
        plt.close("all")

        title = f"Feature Importance Knee Point {suffix}"
        knee_y = round(feature_importance.iloc[knee], 2)
        kn.plot_knee()
        plt.title(title)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.scatter(x=[knee], y=knee_y, c="r")
        plt.text(x=knee+feature_importance.shape[0]/100, y=knee_y+feature_importance.max()/100, 
                s=f"({knee+1}, {round(knee_y, 2)})")
        plt.legend("", frameon=False)
        plt.savefig(output_directory+title+".png", transparent=True, 
                    bbox_inches="tight", dpi=144)
        plt.close("all")

    return selected_features


def select_feature(X:pd.DataFrame, y:pd.DataFrame, dir):
    # Select feature
    # # Adaptive elastic net
    # feature_importance = adaptive_elasticnet(X, Y) 
    # aenet = select_by_knee(feature_importance)
    
    # # Gradient boosting regressor
    # feature_importance = extreme_gradient_boosting(X, Y)
    # xgb = select_by_knee(feature_importance, output_directory=dir, suffix="GBM")
    
    # # Monotonicity
    # def _cal(x):
    #     x = x.dropna()
    #     return np.abs((x>=0).sum()/(x.shape[0]-1)-(x<0).sum()/(x.shape[0]-1))
    # monotonicity = pd.Series([_cal(x) for (_, x) in X.diff().items()], 
    #                          index=X.columns).sort_values()
    # mono = select_by_knee(monotonicity, output_directory=dir, suffix="Monotonicity")
    
    # # Trendability
    # def _cal(x, y):
    #     x = x.dropna()
    #     y = y.loc[x.index]
    #     return np.abs(np.corrcoef(x, y)).min()
    # trendability = pd.Series([_cal(x, y) for (_, x), (_, y) in zip(X.items(), Y.items())], 
    #                    index=X.columns).sort_values()
    # trend = select_by_knee(trendability, output_directory=dir, suffix="Trendability")

    # Set Y to pair X features
    Y = pd.DataFrame(np.tile(y.squeeze().values, (len(X.columns), 1)).T, 
                     index=y.index, columns=X.columns)
    
    # Select by scores
    def select_by(score, suffix):
        score = pd.Series(score, index=X.columns)
        selected = select_by_knee(score, output_directory=dir, suffix=suffix)
        return selected
    selected_features = []
    feature_selection = pd.DataFrame(index=X.columns)
    for (score, suffix) in [
        [get_monotonicity(X), "Monotonicity"], 
        [get_prognosability(X), "Prognosability"], 
        [get_trendability(X, Y=Y), "Trendability"], 
        [get_reliability(X, Y), "Reliability"], 
        # [get_external_robustness(X, Y), "External_robustness"], 
        [get_internal_robustness(X), "Internal_robustness"], 
        ]:
        selected = select_by(score, suffix)
        selected_features.extend(selected)
        feature_selection[suffix] = score
    # Select
    selected_features = list(set(selected_features))
    # selected_features = list(set(xgb) | set(mono) | set(trend))

    # Save
    feature_selection.to_csv(dir+"Feature Selection.csv")

    return selected_features


class XPreprocessor():
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, window_size=1):
        self.columns = X.columns

        # Informative columns
        self.columns = X.columns[X.nunique()>2].tolist()
        X = X[self.columns]

        # Scale
        self.scalar = StandardScaler()
        self.scalar.fit(X)

        # # Smoothen data
        # self.window_size = window_size

    def transform(self, X:pd.DataFrame):
        index = X.index

        # Informative columns
        X = X[self.columns]

        # Scale
        X = pd.DataFrame(self.scalar.transform(X), index=index, columns=self.columns)

        # # Smoothen data
        # X = X.ewm(span=self.window_size).mean()

        return X
    
    def fit_transform(self, X:pd.DataFrame, window_size=1):
        self.fit(X, window_size)
        X = self.transform(X)
        return X
    
    def inverse_transform(self, X):
        # Scale
        X = self.scalar.inverse_transform(X)
        return X
    

class YPreprocessor():
    def __init__(self):
        pass

    def fit(self, Y:pd.DataFrame):
        self.columns = Y.columns

    def transform(self, Y:pd.DataFrame):
        return Y
    
    def fit_transform(self, Y:pd.DataFrame):
        self.fit(Y)
        Y = self.transform(Y)
        return Y
        
    def inverse_transform(self, Y):
        return Y
    

class EPreprocessor():
    def __init__(self):
        pass

    def fit(self, E:pd.DataFrame):
        self.columns = E.columns

    def transform(self, E:pd.DataFrame):
        return E
    
    def fit_transform(self, E:pd.DataFrame):
        self.fit(E)
        E = self.transform(E)
        return E
        
    def inverse_transform(self, E):
        return E
