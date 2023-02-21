import subprocess
import requests

base_url = 'http://127.0.0.1:8000'

response1 = subprocess.run(['curl', base_url+'/prediction'],
                           capture_output=True)

response2 = requests.get(base_url+'/prediction')

print(response1.stdout) # b'[1]'
print(response2.content) # b'[1]'
