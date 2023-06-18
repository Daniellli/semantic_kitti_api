


import os 
import sys

from os.path import join, split,exists, isdir,isfile

sys.path.append(os.getcwd())
from utils.pc_utils import *
from utils import * 


from utils.utils import * 
import numpy as np 


def gen_color_dict(labels):
    unique_labels = np.unique(labels)
    res = {}
    for l in unique_labels:
        res[l] =np.random.randint(0,256,(3)).tolist()
    return res

def map_label_to_pc_color(labels,color_dict):
    
    
    ans = []

    for label  in labels:
        ans.append(color_dict[label].copy())

    return np.array(ans)




def remap_label(labels):    
    unique_labels = np.unique(labels)

    map_dict = {x:idx for idx,x in enumerate(unique_labels)}    
    return np.array([map_dict[x] for x in labels])



def downsample_pc(pc,label ,num = 40000):
    
    choices = np.random.randint(0,len(label),num)
    pc = pc[choices]
    label = label[choices]
    return pc,label



'''
description: for multiple sequence evaluation 
return {*}
'''
class MultiSementicKittiGtLoader:
    

    def __init__(self,root,sequence_list):

        
        loader_list = []
        accumulated_labels = []
        length_list = []
        for sequence in sequence_list:
            tmp_loader = SementicKittiGtLoader(root,sequence)
            length_list.append(tmp_loader.__len__())
            loader_list.append( tmp_loader)
            accumulated_labels += tmp_loader.get_label_list()
        self.length_list = length_list
        self.loader_list = loader_list
        self.accumulated_labels = accumulated_labels
            

    def __len__(self):
        return len(self.accumulated_labels)


    def get_name(self,sample_idx):

        for idx, length in enumerate(self.length_list):
            if length < sample_idx:
                sample_idx -= length
            else:
                target_loader = self.loader_list[idx]
                name = target_loader.idx2name(sample_idx)

        return name 

    
    def __getitem__(self,idx):
        return self.accumulated_labels[idx]


class SementicKittiGtLoader:


    def __init__(self,root,sequence='08'):

        self.root = root
        self.sequence = sequence

        self.seq_root = join(self.root,'sequences',sequence)

        self.label_path = join(self.seq_root,'labels')
        self.labels = sorted([x for x in os.listdir(self.label_path ) if x.endswith('label')])
        
        self.times = np.loadtxt(join(self.seq_root,'times.txt'))
        self.poses = np.loadtxt(join(self.seq_root,'poses.txt'))
        # self.calib = np.loadtxt(join(self.seq_root,'calib.txt'),delimiter='\n')


        self.velodyne_path = join(self.seq_root,'velodyne')

        self.velodynes =sorted( [x for x in os.listdir(self.velodyne_path) if x.endswith('bin')])

        
        
        

    def get_label_list(self):
        ans = []
        for idx in range(self.__len__()):
            ans.append(self.__getitem__(idx))
        return ans 
        

    def __len__(self):
        return len(self.labels)
    

    def get_label(self,idx):
         
        # print(self.labels[idx])
        label = np.fromfile(join(self.label_path,self.labels[idx]), dtype=np.uint32)

        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        # print(np.unique(sem_label, return_counts=True))
        # print(np.unique(inst_label, return_counts=True))


        return sem_label,inst_label
    
    def idx2name(self,idx):

        return self.labels[idx].split('.')[0]
    
    def name2idx(self,name):
        return self.labels.index(name+'.label')
    

    def get_velodyne(self,idx):
        
        # print(self.velodynes[idx])

        scan = np.fromfile(join(self.velodyne_path,self.velodynes[idx]), dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission


        return points,remissions
    

    def save(self,idx):
        #* for debug 
        self.save_dir =  'logs/vis'
        make_dir(self.save_dir)

        label = self.get_label(idx)
        pc ,_= self.get_velodyne(idx)

        label = remap_label(label)

        
        write_ply_color(pc,label,filename=join(self.save_dir,self.idx2name(idx)+".ply"))
        
    def __getitem__(self,idx):
        sem_label,inst_label = self.get_label(idx)

        # return join(self.label_path,self.labels[idx])
        return sem_label
    

        


if __name__ == "__main__":

    print(os.getcwd())
    semantic_gt_loader = SementicKittiGtLoader(join(os.getcwd(),'../semantic_kitti/dataset'))
    print('test done ')
