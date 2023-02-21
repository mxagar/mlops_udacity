"""A simple Flask API/App.

Usually, a Flask app has these minimum steps:

1. Instantiate the Flask app
2. Define the endpoints so that users can interact
3. Run the app with chosen host and port values

To execute the app:

    $ python app_dataset.py

and the app is served. We get the IP where it's served,
but usually, we can always access it via 127.0.0.1
or localhost from our local machine.

To use an endpoint, we run in another terminal:

    $ curl "http://127.0.0.1:8000?user=Mikel"

and we get back True in return.

Or:

    $ curl "http://127.0.0.1:8000/medians?filename=demodata.csv"

and we get back

    year            1990.0
    population    935933.0

Note: demodata.csv should be in the same folder for the previous usage example
"""

from flask import Flask, request
import pandas as pd

app = Flask(__name__)

# We can define so many auxiliary functions as we want
# here or in separate modules, too.
def read_pandas(filename):
    data = pd.read_csv(filename)
    return data

# curl "http://127.0.0.1:8000?user=Mikel"
@app.route('/')
def index():
    user = request.args.get('user')
    return str(user=='Mikel') + '\n'

# curl "http://127.0.0.1:8000/medians?filename=demodata.csv"
# (if demodata.csv is in the same folder)
@app.route('/medians')
def summary():
    filename = request.args.get('filename')  
    data = read_pandas(filename)
    return str(data.median(axis=0))

app.run(host='0.0.0.0', port=8000)
