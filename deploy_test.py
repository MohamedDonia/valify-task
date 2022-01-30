
"""
Created on Tue Jan 25 17:42:26 2022

@author: Mohamed Donia
"""


import requests
import json
import os 


''' post image and return the response '''
data_path = r'./data/asl_alphabet_test/asl_alphabet_test'
test_data = os.listdir(data_path)


addr = 'http://localhost:5000'
service_url = addr + '/get_alphabet'


# set model type as resnet18
model_types = ['resnet18', 'mobilenetv2', 'squeezenet']
d = requests.patch(addr + '/choose_model', json={'model':model_types[2]})
# set device type either 'cpu' or 'cuda'
s = requests.patch(addr + '/set_device', json={'device':'cpu'})


img = open(data_path + '/' + test_data[0], 'rb').read()
r= requests.get(service_url, files={"image": img})
data_file = json.loads(r.text)
print(data_file['char'])

