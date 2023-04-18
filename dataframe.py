
import pandas as pd

df = pd.DataFrame({
    'name': ['A', 'B', 'A', 'D', 'E', 'F'],
    'lecture': ['math', 'math', 'science', 'science', 'science', 'science'],
    'score': [50, 60, 70, 80, 90, 100],
    're': [1, 0, 1, 0, 1, 0]
})

result = df.pivot_table(index='lecture',
                            values=['score', 're', 'name'],
                            aggfunc={'score': 'mean', 're': 'sum', 'name': lambda x: ','.join(x)},
                            margins=True,
                            margins_name='total',
                            dropna=False)

result2 = df.pivot_table(index='lecture',
                            values=['score', 're', 'name'],
                            aggfunc={'score': 'mean', 're': 'sum', 'name': lambda x: ','.join(sorted(set(x)))},
                            margins=True,
                            margins_name='total',
                            dropna=False)

result2
result.reset_index(inplace=True)
result.rename(columns={'index': 'lecture', 'score': 'avg_score', 're': 'total_re'}, inplace=True)
result['name'] = result['name'].str.join(',')