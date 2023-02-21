from flask import Flask, request
import pandas as pd

app = Flask(__name__)

def read_pandas(filename):
    data = pd.read_csv(filename)
    return data

# curl "http://127.0.0.1:8000?user=Mikel"
@app.route('/')
def index():
    user = request.args.get('user')
    return "Hello " + user + '!\n'

# curl "http://127.0.0.1:8000/size?filename=testdata.csv"
@app.route('/size')
def size():
    filename = request.args.get('filename')
    data = read_pandas(filename)
    return str(len(data.index))

# curl "http://127.0.0.1:8000/summary?filename=testdata.csv"
@app.route('/summary')
def summary():
    filename = request.args.get('filename')
    data = read_pandas(filename)
    return str(data.mean(axis=0))

app.run(host='0.0.0.0', port=8000)
