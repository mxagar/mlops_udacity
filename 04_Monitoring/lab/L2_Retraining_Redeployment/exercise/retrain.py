import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data_location = "./sales.csv"
df = pd.read_csv(data_location)

# Transform
X = df.loc[:,['timeperiod']].values.reshape(-1, 1)
y = df['sales'].values.reshape(-1, 1).ravel()

# Instantiate model
lr = LinearRegression()
# Re-Train
model = lr.fit(X, y)

# Persist file with extracted name
deployed_name = "model.pkl"
pickle.dump(model, open('./production/' + deployed_name, 'wb'))
