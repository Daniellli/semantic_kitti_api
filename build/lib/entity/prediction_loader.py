'''
Author: daniel
Date: 2023-03-21 20:54:42
LastEditTime: 2023-03-22 12:29:10
LastEditors: daniel
Description: 
FilePath: /semantic_kitti_api-master/entity/prediction_loader.py
have a nice day
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from astropy import coordinates as ac
import copy
import torch

import os 
from os.path import join, split,exists, isdir,isfile

# file_root =  join(os.getcwd(),'..')
# os.chdir(file_root)
import sys

sys.path.append(os.getcwd())

from utils.pc_utils import *
from utils import * 


from utils.utils import * 
import numpy as np 



class MultiPredictionLoader:

    def __init__(self,model_predition_path,sequence_list) :

        loader_list = []
        for sequence in sequence_list:
            loader_list.append(PredictionLoader(model_predition_path,sequence))
        self.loader_list = loader_list

        
        accumulated_precition = []
        
        for loader in self.loader_list:
            accumulated_precition += loader.get_prediction_list()



        self.accumulated_precition = accumulated_precition


    def is_mapped(self):
        
        return self.loader_list[0].is_mapped()
        
    def __len__(self):
        return len(self.accumulated_precition)
    
    def __getitem__(self,idx):
        
        return self.accumulated_precition[idx]
        



        
    



class PredictionLoader:
    def __init__(self,model_predition_path,sequence='08'):
        
        self.root = join(model_predition_path,'sequences',sequence)
        
        self.pc_pred_path = join(self.root,'point_predict')
        self.pc_pred_list = sorted(os.listdir(self.pc_pred_path))

        self.uncertainty_path = join(self.root,'uncertainty')
        self.uncertainty_list = sorted(os.listdir(self.uncertainty_path))

        self.name_list = [x.split('.')[0] for x in self.uncertainty_list]

        assert len(self.uncertainty_list) == len(self.pc_pred_list )

    def __len__(self):
        return len(self.uncertainty_list)
    
    

    def name2idx(self,name):
        return self.name_list.index(name)
    
    def idx2name(self,idx):
        return self.name_list[idx]

    def getitem(self,idx):


        # return np.fromfile(join(self.pc_pred_path,self.pc_pred_list[idx]), dtype=np.uint32),\
        #      np.fromfile(join(self.uncertainty_path,self.uncertainty_list[idx]), dtype=np.float32)
        
        #? why int32?  not uint32 according to origin evaluation code.
        return np.fromfile(join(self.pc_pred_path,self.pc_pred_list[idx]), dtype=np.int32),\
             np.fromfile(join(self.uncertainty_path,self.uncertainty_list[idx]), dtype=np.float32)
        
    '''
    description: for evaluation 
    param {*} self
    param {*} idx
    return {*}
    '''
    def __getitem__(self,idx):
        predictions,scores=self.getitem(idx)

        predictions = predictions.reshape((-1)) & 0xFFFF 
        scores = scores.reshape((-1))


        
        # return join(self.pc_pred_path,self.pc_pred_list[idx]),\
        #         join(self.uncertainty_path,self.uncertainty_list[idx])
    
        return predictions,scores
                

    def get_prediction_list(self):
        
        ans = []
        for idx in range(self.__len__()):
            ans.append(self.__getitem__(idx))
        return ans
        
        
    def is_mapped(self):
        pc_pred ,uncertainty= self.getitem(0)
        print(np.unique(pc_pred),len(np.unique(pc_pred)))
        
        # print(np.unique(pc_pred),len(np.unique(pc_pred)))
        return pc_pred.max() >19
        
        

    

if __name__ == "__main__":


    prediction_root = '/data1/liyang/semantic_kitti_api-master/datasets/predictions/sequences/08/n19_four_losses_with_shapenet_anomaly'


    epoch_names= sorted(os.listdir(prediction_root),key=lambda k: int(k.split('_')[-1]))

    for epoch_name in epoch_names:
        loader = PredictionLoader(join(prediction_root,epoch_name))
        if loader.is_mapped():
            print(epoch_name,'is mapped ')
        else :
            print(epoch_name,'is not  mapped ')


    


