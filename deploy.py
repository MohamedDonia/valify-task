
"""
Created on Tue Jan 25 17:18:18 2022

@author: Mohamed Donia
"""

import json 
import numpy as np
import cv2
import os 
import sys
import torch
sys.path.append(os.path.abspath("pytorch-image-models"))
from timm import create_model
from torchvision.models.squeezenet import squeezenet1_0
from flask import Flask, request, Response
import jsonpickle




# read dictionary file
with open(r'alphabet dictionary.json', 'r') as file:
    alpha_dict = json.load(file)
alpha_dict_inv = {k:v for v,k in alpha_dict.items()}

# read test data file 
test_data = os.listdir(r'./data/asl_alphabet_test/asl_alphabet_test')


# set default paramreters :
    
# device :
global DEVICE
DEVICE = 'cpu'
IMG_SIZE = (224, 224)

# models :
global model
model = squeezenet1_0(pretrained=False,
                      num_classes = 29)
model.load_state_dict(torch.load(r'squeeze net weights/squeezenet1_0_1.pt', 
                                 map_location=DEVICE))
model.to(DEVICE)
model.eval()



def predict_alphabet(model, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img/255.0
    img = img.transpose(2,1,0)
    img = np.expand_dims(img, axis = 0)
    img = torch.tensor(np.float32(img)).to(DEVICE)
    pred = model(img)
    pred = torch.nn.functional.softmax(pred, dim=1)
    pred  = torch.argmax(pred, dim=1).cpu().item()
    return alpha_dict_inv[pred]




alphabet_flask = Flask(__name__)
@alphabet_flask.route('/get_alphabet', methods=['GET'])
def get_alphabet():
    # read image
    data = request.files['image']
    # convert binary :
    nparr = np.frombuffer(data.read(), np.uint8)  
    # decode image :
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)    
    # processing :
    character = predict_alphabet(model, img)
    response = {
        'status': 'ok',
        'char': character
        }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled,
                    status=200)
   
@alphabet_flask.route('/choose_model', methods=['PATCH'])
def choose_model():
    data = request.json['model']
    if data == 'resnet18':
        message  = 'Loading Resnet18 done..'
        model = create_model(model_name = 'resnet18', 
                         pretrained=False,
                         num_classes = 29)
        model.load_state_dict(torch.load(r'resnet18 weights/resnet18_1.pt', 
                                 map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    elif data == 'mobilenetv2':
        message  = 'Loading MobileNetV2 done..'
        model = create_model(model_name = 'mobilenetv2_100', 
                         pretrained=False,
                         num_classes = 29)
        model.load_state_dict(torch.load(r'mobilenetv2 weights/mobilenet_1.pt', 
                                 map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    elif data == 'squeezenet':
        message  = 'Loading SqueezeNet done..'
        pass
        
    else:
        message = 'Enter Valid Model ...'
    
    
    response = {'message': message}    
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled,
                    status=200)
    
@alphabet_flask.route('/set_device', methods=['PATCH'])
def set_device():
    data = request.json['device']
    if data=='cpu':
        pass
    elif data=='cuda':
        DEVICE = 'cuda:0'
    
    return Response(response={},
                    status=200)





# start flask app
if __name__ == '__main__':
    alphabet_flask.run(host="0.0.0.0", 
                       port=5000)
    
