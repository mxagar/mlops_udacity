"""This simple script requires to have an API
with the specified endpoint running. For instance,
./app.py.

To use this, run in one terminal

    $ python app.py
    
Then, run in another terminal:

    $ python app_request.py
    
"""
import requests
import subprocess

endpoint_url = "http://127.0.0.1:8000/hello?user=Mikel"

response = requests.get(endpoint_url) # GET method
print(response.content) # extract answer: b'Hello Mikel!'

response = subprocess.run(["curl", endpoint_url], capture_output=True)
print(response.stdout) # extract answer: b'Hello Mikel!'
