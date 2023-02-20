import ast
import pandas as pd

with open('historicmeans.txt', 'r') as f:
    means_list = ast.literal_eval(f.read())

the_data=pd.read_csv('samplefile2.csv')
the_means=list(the_data.mean())
print(the_means)
# [3.0, 4.0, 0.5]

mean_comparison=[(the_means[i]-means_list[i])/means_list[i] for i in range(len(means_list))]
print(mean_comparison)
# [0.0, 0.29032258064516125, -0.16666666666666663]

nas=list(the_data.isna().sum()/len(the_data))
print(nas)
# [0.25, 0.25, 0.0]
