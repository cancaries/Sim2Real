# iter scene_number from 0 to 9
for i in {0..9}; do
    CUDA_VISIBLE_DEVICES=6 python render_single_frame.py --config ./configs/example/waymo_render_096.yaml mode novel scene_number $i
done