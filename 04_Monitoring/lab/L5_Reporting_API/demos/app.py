"""A simple Flask API/App.

Usually, a Flask app has these minimum steps:

1. Instantiate the Flask app
2. Define the endpoints so that users can interact
3. Run the app with chosen host and port values

To execute the app:

    $ python app.py

and the app is served.

Then, to use the endpoint, in another terminal:

    $ curl 127.0.0.1:8000?number=5

and we get back 6 in return.

"""

from flask import Flask, request

# 1. Instantiate Fask app
app = Flask(__name__)

# 2. Define the endpoints with .route()
# The default enpoint is "/"
@app.route('/')
def index():
    # We get an input from the user wth requests -> number
    # We return the input +1 as a string
    number = request.args.get('number')
    return str(int(number)+1)+'\n'

# More endpoints
# ...

# 3. Run the app
# host=0.0.0.0: the app should work in whatever IP
# is assigned to our server (it's like a placeholder)
# port=8000: the port where the app is communicating;
# common port in API configuration
app.run(host='0.0.0.0', port=8000)
