import ast
import pandas as pd

with open('healthdata.txt', 'r') as f:
    means_list = ast.literal_eval(f.read())
    
the_data=pd.read_csv('bloodpressure.csv')
the_means=list(the_data.mean())

mean_comparison=[(the_means[i]-means_list[i])/means_list[i] for i in range(len(means_list))]
print(mean_comparison)
# [-0.08285714285714281, -0.26710526315789473, -0.06451612903225806]

nas=list(the_data.isna().sum())
print(nas)
# [0, 1, 2]