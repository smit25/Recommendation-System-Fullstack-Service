import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sqlalchemy import create_engine
from utils.sql import get_engine

WINDOW_FREQ  = '7D'        
CORR_THRESH  = 0.8         
GINI_INCR    = True 

engine = get_engine()
df = pd.read_sql(
    "SELECT timestamp, user_id, recommendations FROM recommend",
    engine,
    parse_dates=['timestamp']
)

df['item_id'] = df['recommendations'].str.split(',')
df = df.explode('item_id')
df['item_id'] = df['item_id'].str.strip()

df.set_index('timestamp', inplace=True)

counts = (
    df
    .groupby([pd.Grouper(freq=WINDOW_FREQ), 'item_id'])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)
corrs = []
for i in range(1, len(counts)):
    prev = counts.iloc[i-1].values
    curr = counts.iloc[i].values
    if prev.std() > 0 and curr.std() > 0:
        corr, _ = pearsonr(prev, curr)
        corrs.append(corr)
    else:
        corrs.append(np.nan)

def gini_coefficient(arr: np.ndarray) -> float:
    arr = np.sort(arr[arr >= 0])
    if arr.sum() == 0:
        return 0.0
    n = len(arr)
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * arr) / (n * arr.sum())) - (n + 1) / n

ginis = counts.apply(lambda row: gini_coefficient(row.values), axis=1)

loop_via_corr = any(c > CORR_THRESH for c in corrs if not np.isnan(c))

loop_via_gini = (ginis.iloc[-1] > ginis.iloc[0]) if GINI_INCR else False


print("Window start dates:", counts.index.tolist())
print("Concentration (ginis):")
for i in ginis.tolist():
    print(i)

