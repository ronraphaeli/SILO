python silo_sample.py \
 --start_from_ckpt=trained_ckpt/RG/sr_4_cnn_ffhq_operator.pt \
 --task_config=configs/super_resolution_4_config.yaml \
 --sigma_condition --idx_image=17 --clamp --diff_model=sd1.5 \
 --prompt="A high quality photo" --scale=0.75 --model=preserve_LatentDegradationNetwork --dataset_name=coco
