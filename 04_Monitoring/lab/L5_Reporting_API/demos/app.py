"""A simple Flask API/App.

Usually, a Flask app has these minimum steps:

1. Instantiate the Flask app
2. Define the endpoints so that users can interact
3. Run the app with chosen host and port values

To execute the app:

    $ python app.py

and the app is served. We get the IP where it's served,
but usually, we can always access it via 127.0.0.1
or localhost from our local machine.

To use an endpoint, we run in another terminal:

    $ curl "http://127.0.0.1:8000?number=5"

and we get back 6 in return.

Or:

    $ curl "http://127.0.0.1:8000/hello?user=Mikel"

and we get back "Hello Mikel!"
"""

from flask import Flask, request

# 1. Instantiate Fask app
app = Flask(__name__)

# 2. Define the endpoints with .route()
# The default enpoint is "/"
@app.route('/')
def index():
    # We get an input from the user with requests -> number
    # We return the input +1 as a string
    # Usage:
    # curl "http://127.0.0.1:8000?number=5"
    number = request.args.get('number')
    return str(int(number)+1)+'\n'

# Another endpoint
@app.route('/hello')
def hello():
    # We get an input from the user with requests -> user
    # We return Hello + input
    # Usage: 
    # curl "http://127.0.0.1:8000/hello?user=Mikel"
    user = request.args.get('user')
    return f"Hello {user}!"

# More endpoints
# ...

# 3. Run the app
# host=0.0.0.0: the app should work in whatever IP
# is assigned to our server (it's like a placeholder)
# port=8000: the port where the app is communicating;
# common port in API configuration
app.run(host='0.0.0.0', port=8000)
