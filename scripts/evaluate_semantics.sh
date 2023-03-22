
###
 # @Author: daniel
 # @Date: 2023-03-20 22:34:26
 # @LastEditTime: 2023-03-22 12:27:37
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /semantic_kitti_api-master/scripts/evaluate_semantics.sh
 # have a nice day
### 


prediction=datasets/predictions/Jan19_four_losses_with_shapenet_anomaly/model_epoch_39;


# python evaluate_semantics.py --dataset ../semantic_kitti/dataset \
# --predictions $prediction --split valid \
# 2>&1 | tee -a logs/eval39.log



python entity/semantic_evaluator.py --dataset datasets/dataset \
--predictions $prediction --split valid \
2>&1 | tee -a logs/new_eval_code.log
