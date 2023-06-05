
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
titanic = sns.load_dataset('titanic')
titanic.head()

# (1) IQR & 이상치

"quantile" in list(dir(pd.DataFrame))

Q1 = titanic['age'].quantile(0.25)
Q3 = titanic['age'].quantile(0.75)
IQR = Q3-Q1
outlier1, outlier2 = Q1-1.5*IQR, Q3+1.5*IQR

# (2) 정렬 : df.sort_values(ascending="")
# (3) df에서 np 사용하기

titanic[titanic['fare']-np.floor(titanic['fare']) != 0 ]

np.round(5.5), np.ceil(5.5) , np.floor(5.5), np.trunc(5.5)
np.round(-5.5), np.ceil(-5.5) , np.floor(-5.5), np.trunc(-5.5)


### 2과목
# - 머신러닝 작업 능력 (전처리,모형 구축,평가)
# - 주요 패키지 : pandas, sklearn

### 3과목
# - 통계 작업 능력 (가설검정, 분산분석, t-test 등)
# - 주요 패키지 : scipy.stats

