#import ast
import pandas as pd
#import numpy as np

# New fictional scores
recent_r2 = 0.55
recent_sse = 49999

# Load previous model scores
previous_scores = pd.read_csv('previousscores_l3demo.csv')

# Increase version: Imagine we have a new model
max_version = previous_scores['version'].max()
this_version = max_version + 1

# Define new score rows
new_row_r2 = {'metric': 'r2', 
              'version': this_version, 
              'score': recent_r2}

new_row_sse = {'metric': 'sse', 
               'version': this_version, 
               'score': recent_sse}

# Append new model score rows
# Optional: Append them ONLY if the model improves the previous ones
# In that case, we would deploy the better model
if recent_r2 > previous_scores.loc[previous_scores['metric'] == 'r2','score'].max():
    previous_scores = previous_scores.append(new_row_r2, ignore_index=True)
    previous_scores = previous_scores.append(new_row_sse, ignore_index=True)
    
# Persist updated scores
previous_scores.to_csv('newscores.csv')
