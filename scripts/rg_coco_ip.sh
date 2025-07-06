python silo_sample.py \
 --start_from_ckpt=trained_ckpt/RG/ip_rg_coco_operator.pt \
 --task_config=configs/center_inpainting_config.yaml \
 --sigma_condition --idx_image=21 --scale=1 --dataset_name=coco \
 --prompt="A high quality photo" --diff_model=sd1.5