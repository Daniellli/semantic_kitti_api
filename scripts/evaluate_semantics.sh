
###
 # @Author: daniel
 # @Date: 2023-03-20 22:34:26
 # @LastEditTime: 2023-03-24 00:30:15
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /semantic_kitti_api-master/scripts/evaluate_semantics.sh
 # have a nice day
### 


# prediction_root=datasets/predictions/Jan19_four_losses_with_shapenet_anomaly/model_epoch_;

start_epoch=0;
total_epoch=10;
# start_epoch=0;
# total_epoch=50;

# for (( i=$start_epoch; i<$total_epoch; i++ ))
# do
#     pred_path=${prediction_root}${i};
#     if test -d $pred_path
#     then
#         echo eval $pred_path;
#         python entity/semantic_evaluator.py --dataset datasets/dataset \
#         --predictions $pred_path --split valid 2>&1 | tee -a logs/new_eval_code_$i.log &
#     fi
# done



python entity/semantic_evaluator.py --dataset datasets/dataset \
--predictions /data1/liyang/semantic_kitti_api-master/model_save_path/model_epoch_19 --split valid \
2>&1 | tee -a logs/new_eval_code.log



#* eval all model 
# python entity/semantic_evaluator.py --dataset datasets/dataset \
# --predictions-root "datasets/predictions/Jan19_four_losses_with_shapenet_anomaly" --split valid \
# 2>&1 | tee -a logs/new_eval_code.log

