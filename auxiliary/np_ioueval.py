#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import sys
import numpy as np

from os.path import join ,exists,split
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
import matplotlib.pyplot as plt





class iouEval:
  def __init__(self, n_classes, ignore=None, writer=None, epoch=None):
    # classes
    self.n_classes = n_classes

    # tensorboard writer
    self.writer = writer
    self.epoch = epoch

    # What to include and ignore from the means
    self.ignore = np.array(ignore, dtype=np.int64)
    self.include = np.array(
        [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
    #!+=====================
    # print("[IOU EVAL] IGNORE: ", self.ignore)
    # print("[IOU EVAL] INCLUDE: ", self.include)
    #!+=====================

    # reset the class counters
    self.reset()

  def num_classes(self):
    return self.n_classes

  def reset(self):
    self.conf_matrix = np.zeros((self.n_classes,
                                 self.n_classes),
                                dtype=np.int64)

    self.unknown_labels = []
    self.unknown_scores = []

  def addBatch(self, x, y, z):  # x=preds, y=targets
    # sizes should be matching
    x_row = x.reshape(-1)  # de-batchify
    y_row = y.reshape(-1)  # de-batchify
    z_row = z.reshape(-1)

    # check
    assert(x_row.shape == x_row.shape)

    # create indexes
    idxs = tuple(np.stack((x_row, y_row), axis=0))

    # make confusion matrix (cols = gt, rows = pred)
    np.add.at(self.conf_matrix, idxs, 1)
    self.unknown_labels.append(y_row)
    self.unknown_scores.append(z_row)

  def getStats(self):
    # remove fp from confusion on the ignore classes cols
    conf = self.conf_matrix.copy()
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = np.diag(conf)
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=0) - tp
    return tp, fp, fn

  def getIoU(self):
    tp, fp, fn = self.getStats()
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union
    iou_mean = (intersection[self.include] / union[self.include]).mean()
    return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

  def getacc(self):
    tp, fp, fn = self.getStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean  # returns "acc mean"
    
  def get_confusion(self):
    return self.conf_matrix.copy()



    
  def get_unknown_indices_one_sample(self,label,uncertainty):
        valid = label != 0 
        
        label = label[valid]
        uncertainty = uncertainty[valid]
        
        
        label[label != 5] = 0
        label[label == 5] = 1
        assert(len(label) == len(uncertainty))
        
        precision, recall, _ = precision_recall_curve(label, uncertainty)#* take long time
        
        aupr_score = auc(recall, precision)
        fpr, tpr, _ = roc_curve(label, uncertainty)
        auroc_score_1 = auc(fpr, tpr)

        # print(f"AUPR: {aupr_score}; \t AUROC: {auroc_score_1}")

        return  {
          "AUPR":aupr_score,
          "AUROC":auroc_score_1,
        }

     


  def get_unknown_indices(self,save_dir):
    self.unknown_labels = np.concatenate(self.unknown_labels)
    self.unknown_scores = np.concatenate(self.unknown_scores)
    valid = self.unknown_labels != 0
    self.unknown_labels = self.unknown_labels[valid]
    self.unknown_scores = self.unknown_scores[valid]
    self.unknown_labels[self.unknown_labels != 5] = 0
    self.unknown_labels[self.unknown_labels == 5] = 1
    assert(len(self.unknown_scores) == len(self.unknown_labels))

    scores_distribution_ood = self.unknown_scores[self.unknown_labels == 1]
    scores_distribution_in = self.unknown_scores[self.unknown_labels != 1]


    scores_distribution_ood.tofile(join(save_dir,"scores_softmax_3dummy_base_ood.score"))
    scores_distribution_in.tofile(join(save_dir,"scores_softmax_3dummy_base_in.score"))
    # print('Save scores distribution successfully!')

    precision, recall, _ = precision_recall_curve(self.unknown_labels, self.unknown_scores)#* take long time
    aupr_score = auc(recall, precision)
    
    

    plt.plot(recall, precision)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("AUPR: " + str(aupr_score))
    plt.savefig(join(save_dir,'AUPR.jpg'))


    plt.figure()
    fpr, tpr, _ = roc_curve(self.unknown_labels, self.unknown_scores)
    plt.plot(fpr, tpr)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.title("AUROC: " + str(auc(fpr, tpr)))
    plt.savefig(join(save_dir,'AUROC.jpg'))
    
    auroc_score_1 = auc(fpr, tpr)
    # auroc_score_2 = roc_auc_score(self.unknown_labels, self.unknown_scores)

    print('AUPR is: ', aupr_score)
    print('AUROC is: ', auroc_score_1)
    print('FPR95 is: ', fpr[tpr > 0.95][0])

    #!+===================================
    accumulated_res  = {
      "OOD/AUPR":aupr_score,
      "OOD/AUROC":auroc_score_1,
      "OOD/FPR95":fpr[tpr > 0.95][0],
      "epoch":self.epoch,
    }
    #!+===================================

    if self.writer is not None and self.epoch is not None:
      self.writer.add_scalar('OOD/AUPR', aupr_score, self.epoch)
      self.writer.add_scalar('OOD/AUROC', auroc_score_1, self.epoch)
      self.writer.add_scalar('OOD/FPR95', fpr[tpr > 0.95][0], self.epoch)

    return accumulated_res





if __name__ == "__main__":
  # mock problem
  nclasses = 2
  ignore = []

  # test with 2 squares and a known IOU
  lbl = np.zeros((7, 7), dtype=np.int64)
  argmax = np.zeros((7, 7), dtype=np.int64)

  # put squares
  lbl[2:4, 2:4] = 1
  argmax[3:5, 3:5] = 1

  # make evaluator
  eval = iouEval(nclasses, ignore)

  # run
  eval.addBatch(argmax, lbl)
  m_iou, iou = eval.getIoU()
  print("IoU: ", m_iou)
  print("IoU class: ", iou)
  m_acc = eval.getacc()
  print("Acc: ", m_acc)
