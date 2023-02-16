import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os

# Get model name from file
# Having the name in a TXT makes possible to parametrize it
# BUT: I would use a config.yaml or sorts...
with open('deployed_model_name.txt', 'r') as f:
    deployed_name = f.read()
print(deployed_name)

# Get data location + filename
with open('data_location.txt', 'r') as f:
    data_location = f.read()
print(data_location)

# Load dataset
df = pd.read_csv(os.getcwd() + data_location)
# Transform
X = df.loc[:,['bed','bath']].values.reshape(-1, 2)
y = df['highprice'].values.reshape(-1, 1).ravel()

# Instantiate model
logit = LogisticRegression(C=1.0, 
                           class_weight=None, 
                           dual=False, 
                           fit_intercept=True, 
                           intercept_scaling=1, 
                           l1_ratio=None, 
                           max_iter=100,
                           multi_class='auto', 
                           n_jobs=None, 
                           penalty='l2',
                           random_state=0, 
                           solver='liblinear', 
                           tol=0.0001, 
                           verbose=0,
                           warm_start=False)
# Re-Train
model = logit.fit(X, y)

# Persist file with extracted name
pickle.dump(model, open('./production/' + deployed_name, 'wb'))