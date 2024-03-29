import pandas as pd
import pickle
from sklearn import metrics
from sklearn.metrics import f1_score

with open('samplemodel.pkl', 'rb') as file:
    model = pickle.load(file)
    
testdata = pd.read_csv('testdata.csv')
X = testdata[['col1','col2']].values.reshape(-1,2)
y = testdata['col3'].values.reshape(-1,1)

predicted = model.predict(X)
print(predicted) # [0 1 0 0]

f1score = metrics.f1_score(predicted,y)
print(f1score) # 0.6666666666666666