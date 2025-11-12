

import os
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns

path = r'C:\Users\kevin\OneDrive - 國立成功大學 National Cheng Kung University (1)\桌面\tb213all.csv'
files = sorted([f.name for f in os.scandir(path) if f.name.endswith('.csv')])

for robot in ['1', '2', '3']:
    data_path = f'C:\Projects\Index Quality\data\{robot}'+os.sep
    os.makedirs(data_path, exist_ok=True)

    df = pd.DataFrame()
    for file in [f for f in files if f.startswith(f'SPT800@T0_robot_{robot}')]:
        df = pd.concat([df, pd.read_csv(os.path.join(path, file), index_col=0)])

    ts = np.array([datetime.timestamp(i) for i in pd.to_datetime(df.index)])

    df = df.sort_index()
    df = df.reset_index(drop=True)
    df.to_csv(data_path+'X.csv')
    
    y_df = pd.DataFrame(ts.max()-ts, columns=['RUL'])
    y_df.to_csv(data_path+'Y.csv')

df = pd.read_csv(path)
df.plot()
np.any(df>20)


dir = r'C:\Users\kevin\Google kevin880987\我的雲端硬碟\Academic\PhD\Projects\Index Quality\data\auo_vibration_of_bearings_tb213\1'+os.sep
path = dir+r'X_extracted.csv'
df = pd.read_csv(path, index_col=0)
X = df.reset_index(drop=True)
Y = pd.DataFrame(X.shape[0]-(np.arange(X.shape[0])), columns=['RUL'])
X.to_csv(dir+'X.csv')
Y.to_csv(dir+'Y.csv')

dir = r'C:\Users\kevin\Google kevin880987\我的雲端硬碟\Academic\PhD\Projects\Index Quality\data storage\AUO\Vib'+os.sep
for sub in os.listdir(dir)[-1:]:
    if not os.path.isdir(dir+sub):
        continue

    # X = pd.DataFrame()
    for i, file in enumerate(np.sort(os.listdir(dir+sub))):
        if not file.endswith('.csv'):
            continue
        df = pd.read_csv(dir+sub+os.sep+file, index_col=0)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.reset_index(drop=True)

        if i==0:
            df.to_csv(dir+sub+os.sep+'X.csv', index=False, mode='a', header=True)
        else:
            df.to_csv(dir+sub+os.sep+'X.csv', index=False, mode='a', header=False)
        # X = pd.concat([X, df])
        del df
    # X = X.reset_index(drop=True)
    # X.to_csv(dir+sub+os.sep+'X.csv', index=True)

    # X = X.reset_index(drop=True).reset_index(drop=False)
    # sns.scatterplot(data=X, x='index', y='1', hue='file', legend=False)
        
