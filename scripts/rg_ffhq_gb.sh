python silo_sample.py \
 --start_from_ckpt=trained_ckpt/RG/gb_rg_ffhq_operator.pt \
 --task_config=configs/gaussian_deblur_config.yaml \
 --sigma_condition --idx_image=1 --clamp