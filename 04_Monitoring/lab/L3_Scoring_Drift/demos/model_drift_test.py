import ast
import numpy as np

new_f1 = 0.38

with open('previousscores_l3demo.txt', 'r') as f:
    f1_list = ast.literal_eval(f.read())

# Test 1: Is the score better than the best so far?
first_test = new_f1 < np.min(f1_list)
print(first_test) # True

# Test 2: Parametric Test: Is the score significantly better than what we've seen so far?
second_test = new_f1 < np.mean(f1_list)-2*np.std(f1_list)
print(second_test) # False

# Test 3: Non-Parametric Test: Is the score significantly better than what we've seen so far?
iqr = np.quantile(f1_list, 0.75)-np.quantile(f1_list, 0.25)
third_test = new_f1 < np.quantile(f1_list, 0.25)-iqr*1.5
print(third_test) # False
