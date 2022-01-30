"""
Created on Sun Jan 23 16:18:27 2022

@author: Mohamed Donia
"""

import pandas as pd 
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm 
import json
import random 
import cv2



class DataPrepare():
    def __init__(self, dir_file = './data'):
        self.dir_file = dir_file + '/asl_alphabet_train/asl_alphabet_train'
        self.files_list = sorted(os.listdir(self.dir_file))
        self.alphabet_dict = {}
        
        self.df_dict = {'path':[], 'label':[]}
        # loop over every file to get the nested images:
        for index, file in enumerate(tqdm(self.files_list)):
            self.alphabet_dict[file] = index
            images = sorted(os.listdir(self.dir_file + '/' + file))
            for im in images:
                self.df_dict['path'].append(self.dir_file + '/' + file + '/' + im)
                self.df_dict['label'].append(index)
                
            
            
        # export dataframe that contain labelling:
        self.df = pd.DataFrame(self.df_dict)
        self.df.to_csv('data.csv', index=False)
        # export json file contain dictionary:
        with open("alphabet dictionary.json", "w") as outfile:
            json.dump(self.alphabet_dict, outfile)
        

def data_vis():
    if not os.path.exists('data.csv'):
        print("dataframe not found , we will run DataPrepare code to prepare data at first .... ")
        dataprepare = DataPrepare()
    # read dataframe 
    dataframe = pd.read_csv('data.csv')
    # read json file
    with open(r'alphabet dictionary.json', 'r') as file:
        data = json.load(file)
        
    inv_dict = {v: k for k, v in data.items()}
    
    random_row = dataframe.sample()
    img = cv2.imread(random_row.iloc[0]['path'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    print(random_row.iloc[0]['path'])
    print(random_row.iloc[0]['label'], inv_dict[random_row.iloc[0]['label']])
    
        
       
        


if __name__=="__main__": 
    # this line of code prepare data so we can use it with pytorch
    # dataprepare = DataPrepare()
    '''
    this line of code will run data visualization function to make sure 
    that we have a correct labels.
    '''
    #data_vis()
    
    