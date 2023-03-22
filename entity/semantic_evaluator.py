#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from os.path import join, split,split, exists, isdir, isfile
# possible splits
splits = ["train", "valid", "test"]
import time
import json

import sys

sys.path.append(os.getcwd())

import wandb 
from tqdm import tqdm

from loguru import logger

from auxiliary.np_ioueval import iouEval

from entity.semantic_kitti_gt_loader import SementicKittiGtLoader,MultiSementicKittiGtLoader

from entity.prediction_loader import MultiPredictionLoader,PredictionLoader



def parse_args():
  parser = argparse.ArgumentParser("./evaluate_semantics.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset dir. No Default',
  )
  parser.add_argument(
      '--predictions', '-p',
      type=str,
      required=None,
      help='Prediction dir. Same organization as dataset, but predictions in'
      'each sequences "prediction" directory. No Default. If no option is set'
      ' we look for the labels in the same directory as dataset'
  )
  parser.add_argument(
      '--split', '-s',
      type=str,
      required=False,
      choices=["train", "valid", "test"],
      default="valid",
      help='Split to evaluate on. One of ' +
      str(splits) + '. Defaults to %(default)s',
  )

  #* numpy by default
  # parser.add_argument(
  #     '--backend', '-b',
  #     type=str,
  #     required=False,
  #     choices= ["numpy", "torch"],
  #     default="numpy",
  #     help='Backend for evaluation. One of ' +
  #     str(backends) + ' Defaults to %(default)s',
  # )

  parser.add_argument(
      '--datacfg', '-dc',
      type=str,
      required=False,
      default="config/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )

  # parser.add_argument(
  #     '--limit', '-l',
  #     type=int,
  #     required=False,
  #     default=None,
  #     help='Limit to the first "--limit" points of each scan. Useful for'
  #     ' evaluating single scan from aggregated pointcloud.'
  #     ' Defaults to %(default)s',
  # )

  
  # parser.add_argument(
  #     '--codalab',
  #     dest='codalab',
  #     type=str,
  #     default=None,
  #     help='Exports "scores.txt" to given output directory for codalab'
  #     'Defaults to %(default)s',
  # )

  # parser.add_argument(
  #     '--prediction_source_folder',
  #     type=str,
  #     required=False,
  #     default=None,
  #     help='.'
  # )
  # parser.add_argument(
  #     '--uncertainty_folder_name',
  #     type=str,
  #     required=False,
  #     default='uncertainty/',
  #     help='.'
  # )
  # parser.add_argument(
  #     '--point_predict_folder_name',
  #     type=str,
  #     required=False,
  #     default='point_predict/',
  #     help='.'
  # )
  # Add tensorboard dir
  # parser.add_argument(
  #     '--tensorboard_runs_dir',
  #     type=str,
  #     required=False,
  #     default='../Open_world_3D_semantic_segmentation/runs',
  #     help='.'
  # )

  FLAGS, unparsed = parser.parse_known_args()

  return FLAGS, unparsed



class SementicEvaluator:


  def __init__(self,args) -> None:


    #* 0.  init logger 

    #* 1. generate gt and prediction loader
    self.args = args

    self.get_data_config()

    self.label_loader =MultiSementicKittiGtLoader(self.args.dataset,self.test_sequences)

    self.prediction_loader = MultiPredictionLoader(self.args.predictions,self.test_sequences)

    #* 2. create evaluator 
    evaluator = iouEval(len(self.data_cfg["learning_map_inv"]), self.ignore, 
                          writer=None, epoch=None)
    evaluator.reset()
    self.evaluator = evaluator




  def get_data_config(self):
      print("Opening data config file %s" % self.args.datacfg)
      DATA = yaml.safe_load(open(self.args.datacfg, 'r'))
      self.data_cfg = DATA

      # get number of interest classes, and the label mappings
      
      class_remap = DATA["learning_map"]
      
      class_ignore = DATA["learning_ignore"]

      maxkey = max(class_remap.keys())
      
      remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
      remap_lut[list(class_remap.keys())] = list(class_remap.values())
      

      self.remap_lut = remap_lut

      # create evaluator
      ignore = []
      for cl, ign in class_ignore.items():
        if ign:
          x_cl = int(cl)
          ignore.append(x_cl)
          print("Ignoring xentropy class ", x_cl, " in IoU evaluation")

      self.ignore = ignore
      # get test set
      self.test_sequences = ['{0:02d}'.format(int(x)) for x in DATA["split"][self.args.split]]

      




  '''
  description: 
  param {*} self
  param {array} args
  param {object} kwds
  return {*}
  #! note: hyper parameter of "limit" and "codelab"  is unused,  so delete
      def limit(label):
        if FLAGS.limit is not None:
          label = label[:FLAGS.limit]  # limit to desired length
  '''
  def __call__(self):
    if not self.prediction_loader.is_mapped():
      logger.info(f"please mapped first ")

      return 
    progress = tqdm(self.label_loader)
    length =self.label_loader.__len__()
    for idx, label in enumerate(progress):
      pred,scores = self.prediction_loader[idx]
    

      valid_index = label != 0 #* 0 is the unlabeled object 

      label = self.remap_lut[label]       # remap to xentropy format
      label = label[valid_index]

      pred = self.remap_lut[pred]       # remap to xentropy format
      pred = pred[valid_index]

      
      scores = scores[valid_index]

      
      self.evaluator.addBatch(pred, label, scores)
      progress.update(idx//length)

    tic = time.time()
    eval_res = self.evaluator.get_unknown_indices(self.args.predictions) #* take long time
    print('spend time :',time.strftime("%H:%M:%S", time.gmtime(time.time() - tic)))
    

    with open(join(self.args.predictions,'anomaly_eval_results.json'),'w') as f :
      json.dump(eval_res,f)
      
    ''' 
    description:   print the sementic evaluation results but unuseful for anomaly detection ?
    param {*} class_jaccard
    param {*} ignore
    param {*} class_strings
    param {*} class_inv_remap
    return {*}
    '''
    def print_eval_results(evaluator):
      # when I am done, print the evaluation
      class_inv_remap = self.data_cfg['learning_map_inv']
      class_strings = self.data_cfg["labels"]
        
      m_accuracy = evaluator.getacc()
      m_jaccard, class_jaccard = evaluator.getIoU()
    
      print('Validation set:\n','Acc avg {m_accuracy:.3f}\n',
            'IoU avg {m_jaccard:.3f}'.format(m_accuracy=m_accuracy,m_jaccard=m_jaccard))
      
      # print also classwise
      for i, jacc in enumerate(class_jaccard):
        if i not in self.ignore:
          print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
              i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))

      # print for spreadsheet
      print("*" * 80)
      print("below can be copied straight for paper table")
      for i, jacc in enumerate(class_jaccard):
        if i not in self.ignore:
          sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
          sys.stdout.write(",")
      sys.stdout.write('{jacc:.3f}'.format(jacc=m_jaccard.item()))
      sys.stdout.write(",")
      sys.stdout.write('{acc:.3f}'.format(acc=m_accuracy.item()))
      sys.stdout.write('\n')
      sys.stdout.flush()


    print('spend  time  : ',time.strftime("%H:%M:%S",time.gmtime(time.time() - tic)))


      

    



if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  
  tic = time.time()


  # print summary of what we will do
  logger.info("*" * 80)
  for k, v in FLAGS.__dict__.items():
      logger.info(f"{k}: {v}")
  logger.info("*" * 80)

  assert(FLAGS.split in splits)


  evaluator = SementicEvaluator(FLAGS)
  evaluator()

  print('total spend  time  : ',time.strftime("%H:%M:%S",time.gmtime(time.time() - tic)))

  
