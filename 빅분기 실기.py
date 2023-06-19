
### 빅데이터분석기사 실기 공부(23년 개정 ver)

import pandas as pd
import numpy as np

### 도움말 사용 방법

# 검색을 할 수 없기 때문에, 파이썬 내부의 도움말 함수를 이용하여야한다 -> help, dir, .__all__
# help는 부연설명이 더 많고, dir은 그 안의 subpackage, module 리스트를 보여준다
# 터미널창으로 읽는 것은 불편하니 메모장에 복사해서 읽기

import scipy
scipy.__all__
dir(scipy)
help(scipy)

# stats가 필요한 모듈인 경우, scipy에서 stats 불러오기

from scipy import stats
dir(stats)
help(stats)

# help의 example을 통해 함수에 필요한 변수, 사용법을 확인
help(stats.ttest_ind)

# https://www.kaggle.com/datasets/agileteam/bigdatacertificationkr
### 1과목
# - DataFrame 처리 능력
# - 주요 패키지 : pandas, numpy

import seaborn as sns
df = sns.load_dataset('titanic')
df.head()

# (1) IQR & 이상치

"quantile" in list(dir(pd.DataFrame))

Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3-Q1
outlier1, outlier2 = Q1-1.5*IQR, Q3+1.5*IQR

# (2) 정렬
# df.sort_values(ascending="")

# (3) df에서 numpy(np) 사용하기

# 소숫점 자리 숫자 있는 데이터 찾기
# 1번 방법
df[df['fare']-np.floor(df['fare']) != 0 ]

# 2번 방법
df[df.fare%1==0]

# 3번 방법
df[df.fare == round(df.age,0)]

np.round(5.5), np.ceil(5.5) , np.floor(5.5), np.trunc(5.5)
np.round(-5.5), np.ceil(-5.5) , np.floor(-5.5), np.trunc(-5.5)

# 결측값 개수 확인
df.isnull().sum()

# 결측값 채우는 코드가 복잡할 경우 inplace 작동 안함
df = sns.load_dataset('titanic')
for sex in df.sex.unique():
    df.loc[df.sex == sex, 'age'].fillna(df.loc[df.sex == sex, 'age'].mean(), inplace=True)

print("inplace 써서 결측값 채웠을 때 결측값 개수 : ", df.age.isnull().sum())

df = sns.load_dataset('titanic')
for sex in df.sex.unique():
    df.loc[df.sex == sex, 'age'] = df.loc[df.sex== sex, 'age'].fillna(df.loc[df.sex == sex, 'age'].mean())

print("inplace 안쓰고 변수에 직접 지정 했을 때  결측값 채웠을 때 결측값 개수 : ", df.age.isnull().sum())

# 결측값 채울 때 map 사용하기 ( 다른 칼럼의 값으로 그룹화 해서 결측값 채우기

df = sns.load_dataset('titanic')
df['age'] = df['age'].fillna(df['sex'].map({'male' : 30,'female' : 25}))

my_dict = {city : y for city, y in zip(df['city'], df['y'])}

# 왜도, 첨도, 로그변환

왜도 = df.fare.skew() # 분포의 기울어진 정도. 왜도가 양수이면 오른쪽꼬리가 길다
첨도 = df.fare.kurt() # 꼬리의 두께. 3보다 크면 정규분포보다 꼬리가 두꺼움
df['log_fare'] = np.log1p(df.fare) # 로그변환

# 특정칼럼의 결측치 제거

df2 = df[~df.fare.isnull()]

# 값 대체하기, replace
df3 = df.copy()
df3['sex'] = df['sex'].replace({'female':'f','male':'m'})
df3['sex']

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

# df = pd.read_csv("file.csv", parse_dates=['Date'], index_col=0)
# 아래 코드를 한줄로 표현함
# df = pd.read_csv("file.csv")
# df['Date'] = pd.to_datetime(df['Date'])
# df = df.set_index('Date')

# 주단위, 2주단위, 월단위로 묶는 방법
df_2 = df.set_index('Date').resample('W').sum() # 1주 : W, 2주 : 2W, 1달 : M

#1일 차이가 나는 시차 특성 만들기
df['previous_PV'] = df['PV'].shift(1)

# n번째 글자 확인법 : .str[:1]

# 특정 칼럼을 기준으로 값마다의 갯수를 알고 싶을때? : df.칼럼명.value_counts()

# 칼럼의 데이터 타입 변경하기 : df = df.astype({'칼럼명' : 'int'})
'''
Numeric Types: int(정수), float(소수), complex(복소수)
Sequence Types: str(문자열), list(리스트), tuple(튜플)
Mapping Type: dict(딕셔너리)
Set Types: set(집합)
Boolean Type: bool(불리언)
Binary Types: bytes, bytearray, memoryview
'''

# 20221201 형태 데이터를 날짜 형식으로 바꾸는 방법
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

# melt 이해하기 -> 풀어쓰기
import pandas as pd
df=pd.DataFrame({'name' : ['A','B','C'], 'Math' : [10,20,30], 'Eng' : [50,60,70]})
pd.melt(df, id_vars=["name"], value_vars=['Math'])

# sigmoid 함수

def sigmoid(x):
    return 1/(1+np.exp(-x))


### 2과목
# - 머신러닝 작업 능력 (전처리,모형 구축,평가)
# - 주요 패키지 : pandas, sklearn

### 3과목
# - 통계 작업 능력 (가설검정, 분산분석, t-test 등)
# - 주요 패키지 : scipy.stats

