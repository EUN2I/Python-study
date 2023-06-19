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

# StandardScaler fit & transform

from sklearn.preprocessing import StandardScaler

data1 = [[0, 2], [0.4, 0.2], [1.1, 10], [11, 19], [34, 21], [6, 40]]
data2 = [[0, 3], [0.4, 0.3], [1.1, 12], [11, 21], [34, 21], [6, 40]]

scaler = StandardScaler()

scaler.fit(data1)
print(scaler.transform(data2)) # data1의 평균, 표준편차로 data2가 정규화됨
print(scaler.fit_transform(data2)) # data2의 평균, 표준편차로 data2가 정규화됨
help(StandardScaler)


print("df.fare.shape : ",df.fare.shape,"df[['fare']].shape : ", df[['fare']].shape)
scaler.fit_transform(df.fare) # 오류 발생
scaler.fit_transform(df[['fare']]) # 정상 작동

'''
pandas 통계 함수

최대값 .max
최소값 .min
평균값 .mean
중앙값 .median
최빈값 .mode
합계 .sum
데이터 수 .count (결측값 제외됨)

분위수 .quantile
분산 .var
표준편차 .std
왜도 .skew
첨도 .kurt

누적합 .cumsum
누적곱 .cumprod
누적 최대값 .cummax
누적 최소값 .cummin

평균의 표준오차 .sem
평균 절대편차 .mad
절대값 .abs
곱 .prod
'''
import sklearn
# sklearn 도움말은 __all__으로 보기
print(dir(sklearn))
print(sklearn.__all__)

# MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['fare'] = scaler.fit_transform(df[['fare']])

# 상위 n개, 하위 n개
df = df.sort_values(by='age', ascending = False)
df.head(10), df.tail(10)


# 날짜 타입 변환 help(pd.Series.dt)
# df['Date'] = pd.to_datetime(df['Date'])
# 새로운 컬럼 추가 (년, 월, 일, 요일)
# df['year'] = df['Date'].dt.year
# df['month'] = df['Date'].dt.month
# df['day'] = df['Date'].dt.day
# df['dayofweek'] = df['Date'].dt.dayofweek


# df 의 lambda 함수 사용법 -> 칼럼에 대한 조건 설정시 axis=1 꼭 넣어야함
# df['Sales_new'] =  df.apply(lambda x: x['Sales'] * 0.8 if x['Events'] == 1 else x['Sales'], axis=1)