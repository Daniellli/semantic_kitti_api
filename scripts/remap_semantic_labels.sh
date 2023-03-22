
###
 # @Author: daniel
 # @Date: 2023-03-20 22:34:26
 # @LastEditTime: 2023-03-22 09:53:18
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /semantic_kitti_api-master/scripts/remap_semantic_labels.sh
 # have a nice day
### 

prediction_root="datasets/predictions/sequences/08/n19_four_losses_with_shapenet_anomaly/model_epoch_";



start_epoch=39;
total_epoch=40;

#* 只能用python 程序解决
for (( i=$start_epoch; i<$total_epoch; i++ ))
do

    pred_path=${prediction_root}${i};

    if test -d $pred_path
    then

        echo processing: $pred_path ;
        
        python entity/remapper.py \
        --predictions $pred_path \
        --split valid --inverse \
        2>&1 | tee -a logs/remap_$i.log

    fi

    #* call all of GPU 
    
done