# Python Dataframe 공부 내용 정리

## Pivot table 복잡한 구조
###   피벗테이블에서 칼럼별 집계방식을 다르게 하고 싶을 때 

* 기본 데이터프레임 : 학생별 시험성적
```commandline
df=pd.DataFrame({'name' : [A,B,A,D,E,F], 'lecture' : ['math','math','science','science','science','science',], 'score' : [50,60,70,80,90,100],'re':[1,0,1,0,1,0]})
* RE : 재수강 여부
```

| NAME | LECTURE | SCORE | RE  |
|------|---------|-------|-----|
| A    | math    | 50    | 1   |
| B    | math    | 60    | 0   |
| A    | science | 70    | 1   |
| D    | science | 80    | 0   |
| E    | science | 90    | 1   |
| F    | science | 100   | 0   |

* 원하는 결과 : 과목별 학생명단, 평균 성적, 재수강 학생수
```commandline
result=pd.DataFrame({'name' : ['A,B','A,D,E,F'], 'score' : [55,85],'re':[1,2]}, index = ['math','science'])
```
| LECTURE | NAME        | SCORE | RE  |
|---------|-------------|-------|-----|
| MATH    | A,B         | 55    | 1   |
| SCIENCE | A,D,E,F     | 85    | 2   |
| TOTAL   | A,B,A,D,E,F | 75    | 3   |

* 피벗테이블 작성코드

```
result = df_re1.pivot_table(index='lecture', 
                            values=['score', 're', 'name'],
                            aggfunc={'score': 'mean', 're': 'sum', 'name': lambda x: ','.join(x)},
                            margins=True, 
                            margins_name='total',
                            dropna=False)

result.reset_index(inplace=True)
result.rename(columns={'index': 'lecture', 'score': 'avg_score', 're': 'sum_re'}, inplace=True)
```
&emsp;&emsp; - aggfunc : 칼럼별 집계방식을 지정 \
&emsp;&emsp;&emsp;&emsp; * sum, mean, median, min, max, count, nunique(중복 제외 고유값 개수) 등 \
&emsp;&emsp;&emsp;&emsp; * lambda 함수 사용 가능 ex) **lambda x: ','.join(x)**  \
&emsp;&emsp; - margins : 총합계 표시 여부, false로 설정하면 총합계 표시 없음 \
&emsp;&emsp; - dropna :  결측값을 계산에 포함할지 여부. false로 설정하면 결측값도 포함 \

* 문자열 합치는 lambda 함수 \
&emsp;&emsp; * 중복값 제거 없이 : lambda x: ','.join(x) \
&emsp;&emsp; * 중복값 제거하는 경우 : lambda x: ','.join(set(x)) \
&emsp;&emsp; * 알파벳순으로 정렬하고 싶다면 sorted 함수 사용 ex) .join(sorted(set(x)))


* 피벗테이블 작성코드 및 결과 (중복값 제거ver.) 
```
result = df_re1.pivot_table(index='lecture', 
                            values=['score', 're', 'name'],
                            aggfunc={'score': 'mean', 're': 'sum', 'name': lambda x: ','.join(sorted(set(x)))},
                            margins=True, 
                            margins_name='total',
                            dropna=False)

result.reset_index(inplace=True)
result.rename(columns={'index': 'lecture', 'score': 'avg_score', 're': 'sum_re'}, inplace=True)
```


| LECTURE | NAME      | SCORE | RE  |
|---------|-----------|-------|-----|
| MATH    | A,B       | 55    | 1   |
| SCIENCE | C,D,E,F   | 85    | 2   |
| TOTAL   | A,B,D,E,F | 75    | 3   |