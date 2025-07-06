python silo_sample.py \
 --start_from_ckpt=trained_ckpt/CNN/sr_8_cnn_ffhq_operator.pt \
 --task_config=configs/super_resolution_8_config.yaml \
 --sigma_condition --idx_image=1 --clamp --model=cnn