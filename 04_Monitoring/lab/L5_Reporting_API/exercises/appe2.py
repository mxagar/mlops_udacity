
from flask import Flask, request
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    user = request.args.get('user')
    return "Hello " + user + '\n'

@app.route('/size')


@app.route('/summary')


app.run(host='0.0.0.0', port=8000)




