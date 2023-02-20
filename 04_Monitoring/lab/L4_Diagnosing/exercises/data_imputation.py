import pandas as pd

the_data = pd.read_csv('samplefile3.csv')
# 
# col1,col2,col3
# 1,2,0
# 4,,0
# 3,2,
# 5,6,1
# ,,0
# 5,,
# ,3,

nas = list(the_data.isna().sum())
na_percents = [nas[i]/len(the_data.index) for i in range(len(nas))]

# pandas.to_numeric: errors=‘coerce’: invalid parsing will be set as NaN
# pandas.mean(skipna=True): default is True
the_data['col1'].fillna(pd.to_numeric(the_data['col1'], errors='coerce').mean(skipna=True), inplace=True)
the_data['col2'].fillna(pd.to_numeric(the_data['col2'], errors='coerce').mean(skipna=True), inplace=True)
the_data['col3'].fillna(pd.to_numeric(the_data['col3'], errors='coerce').mean(skipna=True), inplace=True)

print(the_data)
# 
#    col1  col2  col3
# 0   1.0  2.00  0.00
# 1   4.0  3.25  0.00
# 2   3.0  2.00  0.25
# 3   5.0  6.00  1.00
# 4   3.6  3.25  0.00
# 5   5.0  3.25  0.25
# 6   3.6  3.00  0.25