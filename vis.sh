python visualize.py --sequence 08 \
  --dataset ../semantic_kitti/dataset \
  --offset 3025 \
  --uncertainty ../semantic_kitti/predictions \
  2>&1 | tee -a logs/vis.log


#  --predictions ../semantic_kitti/predictions \