import ast
import numpy as np

newr2=0.3625

with open('previousscores.txt', 'r') as f:
    r2_list = ast.literal_eval(f.read())

# Test 1: Raw Test: Is the score better than the best so far?
first_test = newr2 > np.max(r2_list)
print(first_test) # False

# Test 2: Parametric Test: Is the score significantly better than what we've seen so far?
second_test = newr2 > np.mean(r2_list)+2*np.std(r2_list)
print(second_test) # False

# Test 3: Non-Parametric Test: Is the score significantly better than what we've seen so far?
iqr = np.quantile(r2_list, 0.75)-np.quantile(r2_list, 0.25)
third_test = newr2 > np.quantile(r2_list, 0.75)+iqr*1.5
print(third_test) # False
