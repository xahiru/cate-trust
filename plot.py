import copy
import math
from collections import defaultdict
import os
import matplotlib
import plotly
import seaborn as sns
import pandas as pd

path = './result.csv'

df = pd.read_csv(path, sep=",", names =['userID', 'itemID', 'rating', 'estimate', 'detail'])
df['error'] = abs(df.rating - df.estimate)
# for i in range(10):
#     print(df.detail[0][i])
for i in range(len(df)):
    df.detail[i] = depart(df.detail[i])


#print(len(df))
