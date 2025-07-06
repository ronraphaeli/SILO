python silo_sample.py \
 --start_from_ckpt=trained_ckpt/RG/rand_ip_rg_ffhq_operator.pt \
 --task_config=configs/random_box_inpainting_config.yaml \
 --sigma_condition --idx_image=20 --scale=1
