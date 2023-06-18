import pandas as pd
import numpy as np

import seaborn as sns
df = sns.load_dataset('titanic')
df.head()

# 특정 칼럼 결측치 제거

df.columns

df.isnull().sum()
df_deck결측제거 = df[~df['deck'].isnull()]

df.deck.fillna(method='bfill', inplace = True) # 바로 뒷 값으로 nan 채우기
df.deck.fillna(method ='pad') # 이전 값으로 nan 채우기