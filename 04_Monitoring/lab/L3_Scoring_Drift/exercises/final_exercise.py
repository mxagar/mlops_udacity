import pandas as pd
import pickle
import ast
import numpy as np
from sklearn.metrics import mean_squared_error

## Score model

with open('l3final.pkl', 'rb') as file:
    model = pickle.load(file)
    
testdata = pd.read_csv('testdatafinal.csv')
X = testdata[['timeperiod']].values.reshape(-1,1)
y = testdata['sales'].values.reshape(-1,1)

predicted = model.predict(X)

new_mse = mean_squared_error(predicted,y)
print(new_mse) # 18938.960000000043

## Check drift

with open('l3finalscores.txt', 'r') as f:
    mse_list = ast.literal_eval(f.read())

# Non-Parametric Test: Is the score significantly WORSE
# than what we've seen so far?
iqr = np.quantile(mse_list, 0.75)-np.quantile(mse_list, 0.25)
print(iqr) # 2055.0
drift_test = new_mse > np.quantile(mse_list, 0.75)+iqr*1.5
print(drift_test) # True