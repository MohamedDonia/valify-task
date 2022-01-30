
"""
Created on Mon Jan 24 09:30:39 2022

@author: Mohamed Donia
"""
import json 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os 



# read json file
with open(r'squeeze net weights with mixed up/squeezenet 1_0 training logs.json', 'r') as file:
    data = json.load(file)

# read json file
with open(r'alphabet dictionary.json', 'r') as file:
    alpha_dict = json.load(file)

alpha_dict_inv = {k:v for v,k in alpha_dict.items()}

df = pd.read_csv('data.csv')
df = df.sample(frac=1).reset_index(drop=True)


#define subplots
f, axarr = plt.subplots(4, 3, figsize=(14, 14))
for i in range(12):
    img = cv2.imread(df.iloc[i, 0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    y = alpha_dict_inv[df.iloc[i, 1]]
    
    axarr[i//3,i%3].imshow(img)
    plt.setp(axarr[i//3,i%3], title=y)
    
    
plt.show()
    
    

  
# plot loss    
plt.plot(data['fold 1']['train loss'], label='train loss')
plt.plot(data['fold 1']['val loss'], label='valiation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()
# plot accuracy
plt.plot(data['fold 1']['train acc'], label='train accuracy')
plt.plot(data['fold 1']['val acc'], label='valiation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.show()
    
max_acc = []
for i, data_ in data.items():
    max_acc.append(max(data_['val acc']))
print(np.mean(max_acc))



'''
def MixedUp(img1, img2, alpha):
    if img2.shape != img1.shape:
        h, w, _ = img1.shape
        img2 = cv2.resize(img2, dsize=(h, w))
        img  = img1 * (1 - alpha) + alpha * img2
    return img.astype(int)

img1 = cv2.imread(r'/home/umbra/Work/valify/data/asl_alphabet_test/asl_alphabet_test/Q_test.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread(r'/home/umbra/Work/valify/data/random background/0aacbdb54e853a0a.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img = MixedUp(img1, img2, 0.12)
plt.imshow(img)
plt.show()

'''