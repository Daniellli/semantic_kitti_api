


prediction=../semantic_kitti/predictions;

ep30_pred_path="ng/Open_world_3D_semantic_segmentation/runs/debug_original/model_epoch_30";



python evaluate_semantics.py --dataset ../semantic_kitti/dataset \
--prediction_source_folder $ep30_pred_path --predictions $prediction --split valid \
2>&1 | tee -a logs/eval.log

# --predictions $ep30_pred_path


