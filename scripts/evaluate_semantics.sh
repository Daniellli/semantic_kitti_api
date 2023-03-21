
###
 # @Author: daniel
 # @Date: 2023-03-20 22:34:26
 # @LastEditTime: 2023-03-21 22:00:29
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /semantic_kitti_api-master/scripts/evaluate_semantics.sh
 # have a nice day
### 


prediction=../semantic_kitti/predictions;

ep30_pred_path="ng/Open_world_3D_semantic_segmentation/runs/debug_original/model_epoch_39";


python evaluate_semantics.py --dataset ../semantic_kitti/dataset \
--prediction_source_folder $ep30_pred_path --predictions $prediction --split valid \
2>&1 | tee -a logs/eval40.log

# --predictions $ep30_pred_path


