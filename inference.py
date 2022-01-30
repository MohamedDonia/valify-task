
"""
Created on Mon Jan 24 21:53:43 2022

@author: Mohamed Donia 
"""

import json 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os 
import sys
import torch
import itertools
sys.path.append(os.path.abspath("pytorch-image-models"))
from timm import create_model
from sklearn.metrics import confusion_matrix
import time
from torchvision.models.squeezenet import squeezenet1_0





def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          figsize=(5,4)):
  
    cm_orig = cm.copy()
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    #plt.figure(figsize=figsize)
    fig = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 # if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:d}".format(cm_orig[i, j]),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    accuracy = np.trace(cm_orig) / float(np.sum(cm_orig))
    misclass = 1 - accuracy

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; error={:0.4f}'.format(accuracy, misclass))
    plt.show()




IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS=20
LR=1e-4
DEVICE = 'cuda:0'
NFOLDS = 5

# read dictionary file
with open(r'alphabet dictionary.json', 'r') as file:
    alpha_dict = json.load(file)
alpha_dict_inv = {k:v for v,k in alpha_dict.items()}
# read test data file 
test_data = os.listdir(r'./data/asl_alphabet_test/asl_alphabet_test')


# models :
model = squeezenet1_0(pretrained=False,
                         num_classes = 29)

model.load_state_dict(torch.load(r'squeeze net weights/squeezenet1_0_1.pt', 
                                 map_location=DEVICE))
model.to(DEVICE)
model.eval()


true_labels = []
predicted_labels = []
time_inference = []

for img_id in test_data:
    img = cv2.imread('./data/asl_alphabet_test/asl_alphabet_test/' + img_id)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img/255.0
    img = img.transpose(2,1,0)
    img = np.expand_dims(img, axis = 0)
    img = torch.tensor(np.float32(img)).to(DEVICE)
    
    t1 = time.time()
    pred = model(img)
    pred = torch.nn.functional.softmax(pred, dim=1)
    pred  = torch.argmax(pred, dim=1).cpu().item()
    t2 = time.time()
    time_inference.append(t2-t1)

    
    true_labels.append(alpha_dict[img_id.replace('_test.jpg', '')])
    predicted_labels.append(int(pred))

cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(0, 29)))
plot_confusion_matrix(cm=cm, target_names=list(alpha_dict.keys()), figsize=(14,10))

print(np.mean(time_inference)*1000)