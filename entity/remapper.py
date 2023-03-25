#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import numpy as np
from tqdm import tqdm
from os.path import join, exists, split, isdir,isfile
import sys
sys.path.append(os.getcwd())
# possible splits
splits = ["train", "valid", "test"]

from utils.utils import process_mp

import time



from entity.semantic_kitti_gt_loader import SementicKittiGtLoader 


def parse_args():
  parser = argparse.ArgumentParser("./remap_semantic_labels.py")

  # parser.add_argument(
  #     '--dataset', '-d',
  #     type=str,
  #     required=False,
  #     default=None,
  #     help='Dataset dir. WARNING: This file remaps the labels in place, so the original labels will be lost. Cannot be used together with -predictions- flag.'
  # )
  
  parser.add_argument(
      '--predictions', '-p',
      type=str,
      required=True,
      default=None,
      help='Prediction dir. WARNING: This file remaps the predictions in place, so the original predictions will be lost. Cannot be used together with -dataset- flag.'
  )
  parser.add_argument(
      '--split', '-s',
      type=str,
      required=False,
      default="valid",
      help='Split to evaluate on. One of ' +
      str(splits) + '. Defaults to %(default)s',
  )
  parser.add_argument(
      '--datacfg', '-dc',
      type=str,
      required=False,
      default="config/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--inverse',
      dest='inverse',
      default=False,
      action='store_true',
      help='Map from xentropy to original, instead of original to xentropy. '
      'Defaults to %(default)s',
  )

  FLAGS, unparsed = parser.parse_known_args()

  return FLAGS, unparsed
   



class ReMapper:
  '''
  description:  remap prediction label range to be consistency with gt label range 
  param {*} self
  param {*} args
    prediction: : the prediction result path
    args.datacfg : the data config file path
    args.inverse : is map from gt label range to prediction label range or reverse 
    split: map with subset
  return {*}
  '''  
  def __init__(self,prediction,datacfg='config/semantic-kitti.yaml',
                inverse=False,split='valid',gt_loader =None):
    
    self.split = split
    self.inverse = inverse
    self.datacfg = datacfg
    assert prediction is not None

    
    #* assume: if(FLAGS.predictions is not None) is always right 
    self.prediction_root = prediction
    #* the directory which  all label file in it need to remap 
    self.label_dir_name = "point_predict" 
    

    #* get map dict

    self.get_map_dict()
    #* get prediction 
    self.get_predictions()
    
    #* remap precition one by one 
    if gt_loader is None :
      self.gt_loader = SementicKittiGtLoader('datasets/dataset')
    else:
      self.gt_loader = gt_loader



  def get_map_dict(self):
    # assert split
    assert(self.split in splits)
    DATA = yaml.safe_load(open(self.datacfg, 'r'))

    # get number of interest classes, and the label mappings
    if self.inverse:
      # print("Mapping xentropy to original labels")
      remapdict = DATA["learning_map_inv"]
    else:
      remapdict = DATA["learning_map"]

    # make lookup table for mapping
    maxkey = max(remapdict.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())

    self.remap_lut = remap_lut  
    # get wanted set
    #* get sequence by split, there are 3 split: test, train, valid
    sequence = []
    sequence.extend(DATA["split"][self.split])

    self.sequence = sequence


  def get_predictions(self):

    label_names = []
    for sequence in self.sequence:
      sequence = '{0:02d}'.format(int(sequence))
      
      label_paths = join(self.prediction_root, 'sequences',sequence,self.label_dir_name)

      # populate the label names
      seq_label_names = sorted([os.path.join(dp, f) for dp, dn, fn in os.walk(
                    os.path.expanduser(label_paths)) for f in fn if ".label" in f])
      
      label_names.extend(seq_label_names)

    self.label_names = label_names
  
  def __len__(self):
    return len(self.label_names)
    

  def remap(self,idx):
      label_file = self.label_names[idx]
      
      label = np.fromfile(label_file, dtype=np.uint32)
      label = label.reshape((-1))
      upper_half = label >> 16      # get upper half for instances
      lower_half = label & 0xFFFF   # get lower half for semantics

      if lower_half.max() > 19:
          print('\n \n It has already been mapped. Skip everything in this folder. ')
          return 

      if (lower_half == 0).all():
          print('\n \n It has already been mapped more than once. Skip everything in this folder. ')
          return 

      lower_half = self.remap_lut[lower_half]  # do the remapping of semantics
      label = (upper_half << 16) + lower_half   # reconstruct full label
      label = label.astype(np.uint32)
      label.tofile(label_file)

  def __call__(self,threads=128):
    tic = time.time()

    if self.__len__() != self.gt_loader.__len__() :
      print(" inference process  is uncomplete, please inference again")
      return 
    

    for idx in range(self.__len__()):
      self.remap(idx)
      

    # process_mp(self.remap,list(range(self.__len__())),threads)
    print('spend time :',time.strftime("%H:%M:%S", time.gmtime(time.time() - tic)))
    print("remap done ")



if __name__ == '__main__':
  FLAGS, unparsed = parse_args()

  remapper = ReMapper(FLAGS.predictions,
                      FLAGS.datacfg,
                      inverse=FLAGS.inverse,
                      split=FLAGS.split)
  remapper(256)