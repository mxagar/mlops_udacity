from flask import Flask, request
import pandas as pd
import pickle

# Instantiate app
app = Flask(__name__)
# Load model
with open('deployedmodel.pkl', 'rb') as file:
    model = pickle.load(file)
   
def read_pandas(filename):
    data = pd.read_csv(filename)
    return data

@app.route('/prediction')
def prediction():
    data = pd.read_csv('predictiondata.csv')
    prediction = model.predict(data)
    return str(prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
