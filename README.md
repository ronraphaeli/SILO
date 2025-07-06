# SILO: Solving Inverse Problems with Latent Operators - ICCV 2025

We aim to make SILO as reproducible as possible.
we begin with the contents of this repo, and than, provide some additional insights on SILO, and future work.
If you encounter problems while trying to use the code, open an issue and provide details.
We hope you have an easy setup for SILO.

Content in this file:

0. Folder Structure & Credits
1. Environment Setup
2. Fast Inference
3. Advanced Inference
4. Training

## 0: Folder Structure & Credits

The code in the folder is an amalgam of code from:

- Readout Guidance: https://github.com/google-research/readout_guidance/tree/main
- Diffusion Posterior Sampling: https://github.com/DPS2022/diffusion-posterior-sampling/tree/main
- Diffusers: https://github.com/huggingface/diffusers/tree/main

and our own additions to make them work with SILO.

The code of Readout guidance and diffusers is under Apache 2.0 License, so we did not remove the license notice from files that contained it prior to our work.

The files in the folder are as follows,

`silo_sample.py` - the code used to generate reconstructions

`silo_utils.py` - contains function and classes that are needed for sampling and training

`/scripts` - contains the sampling scripts

`/sample_results` - in this folder the reconstructions will be saved (created after sampling with silo_sample.py)

`calc_batch.py` - calculate the average of the reported metrics in a folder with log files created from silo_sample.py

`/configs` - the configs for the degradation tasks

`/training` - the folder with the training scripts

`/data` - this folder contains small parts of the FFHQ and coco datasets, we give them here for easy inference. you can choose to use the full FFHQ / COCO datasets if you have access to them.

`/trained_ckpt` - contains the checkpoints of the trained operators

The rest of the folders and files are needed for sampling / training and mostly were provided as part of the github repositories we build upon.

## 1: Environment Setup

You can use micromamba / anaconda etc... to create the virtual environment.
(we recommend micromamba for faster setup https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

Run the commands when the working directory is silo.

0. first, create the environment, using

```bash
cd SILO
micromamba env create -n silo -f silo_env.yaml
micromamba activate silo
```

now, we need to download some additional modules.

1. [Download](https://pytorch.org/get-started/previous-versions/#v240) PyTorch version 2.4.0 with appropriate CUDA version
   for example, we use

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

2. run the following installs

```bash
pip install torchmetrics
micromamba install -n silo datasets=2.21.0
pip install iopath
pip install huggingface_hub==0.24.6
pip install accelerate==0.31.0
```

3. Download checkpoints from github:

```bash
wget -L trained_ckpt.zip https://github.com/ronraphaeli/SILO/releases/download/Checkpoints/trained_ckpt.zip
unzip trained_ckpt.zip
rm trained_ckpt.zip
```

4. give permission to run the sampling scripts

```bash
chmod +x ./scripts/*
```

## 2: Fast Inference

```bash
cd SILO
micromamba activate silo
./scripts/{wanted script}
```

Where `{wanted script}` depends on what you want to run:

| name of script     | operator type    | dataset | degradation         |
| ----------------   | ---------------- | ------- | ------------------- |
| rg_coco_ip.sh      | Readout Guidance | COCO    | Inpaint             |
| cnn_coco_sr_4.sh   | CNN              | COCO    | Super Resolution x4 |
| rg_ffhq_gb.sh      | Readout Guidance | FFHQ    | Gaussian blur       |
| rg_ffhq_ip.sh      | Readout Guidance | FFHQ    | Inpaint             |
| rg_ffhq_rand_ip.sh | Readout Guidance | FFHQ    | Random box Inpaint  |
| rg_ffhq_jpeg.sh    | Readout Guidance | FFHQ    | JPEG                |
| rg_ffhq_sr_4.sh    | Readout Guidance | FFHQ    | Super Resolution x4 |
| rg_ffhq_sr_8.sh    | Readout Guidance | FFHQ    | Super Resolution x8 |
| cnn_ffhq_sr_8.sh   | CNN              | FFHQ    | Super Resolution x8 |

## 3: Advanced Inference

Advanced inference let you to check more types of reconstructions

we show the command to run, each `{param}` indicate that you can choose it from a group listed below

```bash
python silo_sample.py \
 --start_from_ckpt={operator_ckpt} \
 --task_config={task_config} \
 --sigma_condition \
 --clamp \
 --idx_image={idx_image} \
 --scale={scale} \
 --dataset_name={dataset_name} \
 --prompt={prompt} \
 --pretrained_model_name_or_path={diff_prior} \
 --noise_sigma={noise_sigma} \
 --model={model} \
 --steps={steps} \
```

- Choose `{operator ckpt}` from one of the checkpoints provided in `/trained_ckpt`
- Choose `{task config}` from one of the configs in `/configs`
- `{idx_image}` should be an int, indicating the index of the image from the dataset
- `{scale}` is the strength of the guidance, also called eta in SILO paper
- `{dataset name}` is the name of the dataset you want to sample. By default, you can choose from `{"coco", "ffhq"}`. But you can add more datasets.
- `{prompt}` is the prompt to sample with. We recommend to use `"A high quality photo of a face"` in case the dataset is ffhq, and `"A high quality photo"` in case the dataset is COCO.
- `{diff prior}` is the diffusion prior used. We follow the conventions of huggingface. You can choose from `{"stablediffusionapi/realistic-vision-v51","botp/stable-diffusion-v1-5"}`. The first is better for FFHQ and the latter is better for COCO.
- `{noise_sigma}` is the amount of noise to add to the measurement. Note that the true amount is half since we add the noise after the images are normalized to be in the range [-1,1]. So if you want to add noise with sigma=0.01, you should set `noise_sigma=0.02`.
- `{model}` indicates whether you want to use a readout guidance or cnn operator. choose from `{"rg", "cnn"}`.
- `{steps}` is the number of steps of the diffusion process. For the results in the paper, we use 999, but we've seen that 500 and even less works as well, though you might need to use a bigger scale for better consistency to the measurement.

To sample a batch of images, you have 2 options for maximum convenience.

1. To sample indexes in a range (for example the images from 46 to 387), replace `--idx_image={idx_image}` with `--start={start} --end={end}`. where `{start}` is the first idx to sample and `{end}` is the first idx to not sample. for example --start=0 --end=1000.
2. To sample specific indexes in a dataset, you can use `--list_to_do={list_to_do}`, where `{list_to_do}` is a list of the indexes to sample with, for example: `--list_to_do="[1,5,78,367,0,3]"`

In both cases, this force the sampling script to accept a `--folder_path={folder_path}`, so the results of the batch sampling will not get mixed up with the fast inference results. We recommend to open another directory for batch sampling results.

## 4: Training

For training, go into the training folder,

```bash
cd training
```

To train a readout guidance network, use `silo_rg_train.py`:

```bash
python silo_rg_train.py --config_path train_rg_silo_config.yaml --notes={notes_for_run,optional} --train_with_noise --l1_loss --wandb_mode={wandb_mode} --train_batch_size={choose_batch_size} --task_config=../configs/{degredation}_config.yaml --dataset_name={dtaset_name}

more options are available in the config train_rg_silo_config.yaml
```

To train a cnn network, use `silo_cnn_train.py`:

```bash
python silo_cnn_train.py --notes={notes for run, optional} --learning_rate={learning_rate} --pretrained_model_name_or_path {diff_prior} --train_with_noise --l1_loss --sigma_condition --task_config=../configs/{degredation}_config.yaml --clamp --wandb_mode={wandb_mode}

more options are available in the arguments
```

{wandb_mode} is chosen from `disabled`: will not use wandb at all  \ `online`: will log the training to wandb. In that case, put your wandb entity in the config file.

Note that for training, you must use some dataset that is not provided in the zip file (because of file size).
You can download FFHQ from `Ryan-sjtu/ffhq512-caption` or download LSDIR from `danjacobellis/LSDIR`.
Just replace the dataset loading to `load_dataset` instead of `load_from_disk` with one of these. for example use `load_dataset('Ryan-sjtu/ffhq512-caption')`, after adding `from datasets import load_dataset`.
