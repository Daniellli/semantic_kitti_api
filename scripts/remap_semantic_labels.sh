







# aa='../semantic_kitti/predictions';
ep30_pred_path="../semantic_kitti/predictions/sequences/08/ng/Open_world_3D_semantic_segmentation/runs/debug_original/model_epoch_30";



python remap_semantic_labels.py --predictions $ep30_pred_path \
--split valid --inverse 2>&1 | tee -a logs/remap.log