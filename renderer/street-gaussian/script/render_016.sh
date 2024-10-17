# iter scene_number from 0 to 9
for i in {0..9}; do
   CUDA_VISIBLE_DEVICES=0 python render_single_frame.py --config ./configs/example/waymo_train_016.yaml mode novel scene_number $i
done

# CUDA_VISIBLE_DEVICES=10 python render_single_frame.py --config ./configs/example/waymo_train_016.yaml mode novel scene_number 0
