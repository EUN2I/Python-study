
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

df[df['fare']-np.floor(df['fare']) != 0 ] # 소숫점 자리 숫자 있는 데이터 찾기

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



### 2과목
# - 머신러닝 작업 능력 (전처리,모형 구축,평가)
# - 주요 패키지 : pandas, sklearn

### 3과목
# - 통계 작업 능력 (가설검정, 분산분석, t-test 등)
# - 주요 패키지 : scipy.stats

