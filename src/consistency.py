

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
from xgboost import XGBRegressor
from aenet import AdaptiveElasticNet


def adaptive_elasticnet(X, Y):
    '''
    The function fits an adaptive elastic net model to the
    input data and returns a sorted series of feature importances.
    
    Args:
      X (pandas.DataFrame): The input features as a pandas DataFrame.
      Y (pandas.DataFrame): The target variable or dependent variable in the dataset.
    
    Returns:
      pandas.Series: A Pandas Series object containing the feature importance values for each
    feature in the input data X, as determined by the Adaptive Elastic Net
    algorithm. The feature importance values are sorted in descending order.
    '''
    aenet = AdaptiveElasticNet().fit(X, Y.squeeze())

    feature_importance = pd.Series(aenet.coef_, name='aenet', index=X.columns)
    feature_importance = feature_importance.sort_values(ascending=False) # sort importance in descending order
    
    return feature_importance


def extreme_gradient_boosting(X, Y):
    '''
    The function performs extreme gradient boosting feature selection on input data
    and returns a sorted list of feature importances.
    
    Args:
      X (pandas.DataFrame): The input features as a pandas DataFrame.
      Y (pandas.DataFrame): The target variable or dependent variable in the dataset.
    
    Returns:
      pandas.Series: A pandas Series object containing the feature importances calculated using
    extreme gradient boosting algorithm.
    '''
    xgb = XGBRegressor(
        nthread=6, 
        )
    xgb = xgb.fit(X, Y)

    feature_importance = pd.Series(
        xgb.get_booster().get_score(importance_type='gain'), 
        name='xgb', 
        )
    feature_importance = feature_importance.sort_values(ascending=False) # sort importance in descending order
    
    return feature_importance


def select_by_knee(feature_importance, 
                   params={'curve': 'convex', 'direction': 'decreasing'}, 
                   min_features=1, ):
    if feature_importance.shape[0]<=1:
        return list(feature_importance.index)

    kn = KneeLocator(
        x=range(feature_importance.shape[0]), 
        y=feature_importance.values.ravel(), 
        **params, 
        )
    if kn.knee:
        knee = min(max(kn.knee, min_features-1), feature_importance.shape[0]-1)
    else:
        knee = feature_importance.shape[0]-1
    selected_features = feature_importance.index[: knee+1] # *5

    return selected_features


def select_feature(X, Y):
    valid = np.all((np.all(X==X, axis=1), np.all(Y==Y, axis=1)), axis=0)
    X, Y = X.iloc[valid], Y.iloc[valid]
    X.loc[:, :] = MinMaxScaler().fit_transform(X)
    Y.loc[:, :] = MinMaxScaler().fit_transform(Y)

    # DataFrame to hold selection results
    selection = pd.DataFrame(columns=X.columns)
    
    # Adaptive elastic net
    feature_importance = adaptive_elasticnet(X, Y)
    selected_features = select_by_knee(feature_importance)
    selection = pd.concat([selection, 
                           pd.Series([1]*len(selected_features), name='aenet', 
                                     index=selected_features).to_frame().T], axis=0)

    # Gradient boosting regressor
    feature_importance = extreme_gradient_boosting(X, Y)
    selected_features = select_by_knee(feature_importance)
    selection = pd.concat([selection, 
                           pd.Series([1]*len(selected_features), name='xgb', 
                                     index=selected_features).to_frame().T], axis=0)

    selection = selection.fillna(0)
    return selection


# def inter_rater_agreement(agreement):
#     j = []
#     for (_, a), (_, b), in combinations(agreement.iterrows(), 2):
        # p0 = a.values==b.values
#         j.extend(a.values * b.values)
#     j = np.nanmean(j)

#     return j


def jaccard(agreement):
    j = []
    for (_, a), (_, b), in combinations(agreement.iterrows(), 2):
        j.extend(a.values * b.values)
    j = np.nanmean(j)

    return j


def get_consistency(agreement):
    assert agreement.shape[0]>1
    j = jaccard(agreement)

    return j