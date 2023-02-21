"""This simple script requires to have an API
with the specified endpoints running, i.e.,
./appe3.py

To use this, run in one terminal

    $ python appe3.py
    
Then, run in another terminal:

    $ python call_api_endpoint.py
    
"""
import requests
import subprocess

def run_request(endpoint_url):
    response = requests.get(endpoint_url) # GET method
    print(response.content)
    response = subprocess.run(["curl", endpoint_url], capture_output=True)
    print(response.stdout)

base_url = "http://127.0.0.1:8000"

endpoint_url = base_url + "?user=Mikel"
run_request(endpoint_url)

endpoint_url = base_url + "/size?filename=testdata.csv"
run_request(endpoint_url)

endpoint_url = base_url + "/summary?filename=testdata.csv"
run_request(endpoint_url)
