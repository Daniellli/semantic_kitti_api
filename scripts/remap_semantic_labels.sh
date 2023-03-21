

prediction_root="datasets/predictions/sequences/08/n19_four_losses_with_shapenet_anomaly/model_epoch_";





start_epoch=0;
total_epoch=100;

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