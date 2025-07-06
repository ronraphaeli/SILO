#this file is based upon https://github.com/google-research/readout_guidance/blob/main/readout_training/train_spatial.py
# we modified the code, but for license, leave the comment below as is from google github. 

# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import torch
from tqdm import tqdm
import wandb
import math
import random
import numpy as np
import sys
import os
sys.path.append("../")
from util.img_utils import  mask_generator

# from readout_training.dataset import ControlDataset
import train_helpers 
# coding=utf-8
import torchvision.transforms as transforms
from silo_utils import load_yaml, DegradationModel, decode, embed_timestep, decode
from guided_diffusion.measurements import get_noise, get_operator
from diffusers import AutoencoderKL
from torchvision.transforms.functional import to_pil_image , to_tensor , normalize , resize
from torchvision.transforms import InterpolationMode
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from datasets import load_from_disk
device = "cuda"

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        # "botp/stable-diffusion-v1-5" or "stablediffusionapi/realistic-vision-v51" ,
        default="stablediffusionapi/realistic-vision-v51" ,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ffhq",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="",
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=1000, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=True,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    # parser.add_argument(
    #     "--adam_beta1",
    #     type=float,
    #     default=0.9,
    #     help="The beta1 parameter for the Adam optimizer.",
    # )
    # parser.add_argument(
    #     "--adam_beta2",
    #     type=float,
    #     default=0.999,
    #     help="The beta2 parameter for the Adam optimizer.",
    # )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help=("the notes to be doc in the wandb run."),
    )
    parser.add_argument(
        "--start_from_ckpt",
        type=str,
        default=None,
        help=("the notes to be doc in the wandb run."),
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default="configs/gaussian_deblur_config.yaml",
        help=("the notes to be doc in the wandb run."),
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help=(
            "the wandb mode"
    ))
    parser.add_argument(
        "--noise_sigma",
        type=float,
        default=None,
        help="noise for dps noiser",
    )
    parser.add_argument(
        "--blur_sigma",
        type=float,
        default=None,
        help="noise for dps noiser",
    )
    parser.add_argument(
        "--sr_factor",
        type=int,
        default=None,
        help="noise for dps noiser",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="noise for dps noiser",
    )
    parser.add_argument(
        "--train_with_noise",
        default=False,
        action="store_true",
        help="noise for dps noiser",
    )
    parser.add_argument(
        "--use_denoiser",
        default=False,
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--l1_loss",
        default=False,
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--non_uniform_t",
        default=False,
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--make_w_t",
        default=False,
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--sr_legacy",
        default=False,
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--jpeg_qf",
        type=int,
        default=None,
        help="noise for dps noiser",
    )
    parser.add_argument(
        "--clamp",
        default=False,
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--cond_inpaint",
        default=False,
        action="store_true",
        help="",
    )
    parser.add_argument("--config_path", type=str)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
# ====================
#     Dataloader
# ====================

# ====================
#        Loss
# ====================
def loss_spatial(aggregation_network, pred, target, sparse_loss=False, control_range=None,args=None):
    target = train_helpers.standardize_feats(pred, target)
    if sparse_loss and control_range is not None:
        loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
        min_value = control_range[0]
        is_zero = target == min_value
        if is_zero.sum() > 0:
            loss = (loss[target == min_value].mean() + loss[target != min_value].mean()) / 2
        else:
            loss = loss.mean()
    else:
        if args.l1_loss:
            loss = torch.nn.functional.l1_loss(pred, target)
        else:
            loss = torch.nn.functional.mse_loss(pred, target)
    return loss

# ====================
#  Validate and Train
# ====================
def validate(config, diffusion_extractor, aggregation_network, dataloader, split, step, run_name,operator,noiser,denoiser,inpaint_mask=None,mask_gen=None,doing_conditional_inpaint=None):
    device = aggregation_network.device
    sparse_loss = config["dataset_args"]["sparse_loss"]
    control_range = config["dataset_args"]["control_range"]
    total_loss = []
    for j, batch in enumerate(dataloader):
        with torch.no_grad():
            
            x = batch['pixel_values'][0:3].to("cuda",dtype=diffusion_extractor.model.dtype)

            if mask_gen is not None:
                inpaint_mask = mask_gen(x)
                y = operator.forward(x, mask=inpaint_mask)
            else: 
                y = operator.forward(x)

            if doing_conditional_inpaint:
                cond_inpaint = inpaint_mask[:,0:1,:,:]
                cond_inpaint = resize(cond_inpaint,(64,64)) #convert it to latent spatial dim
            else:
                cond_inpaint = None

            # if inpaint_mask is not None:
            #     # Forward measurement model (Ax + n)
            #     if not args.cond_inpaint:
            #         y = operator.forward(x, mask=inpaint_mask)
            #     if args.cond_inpaint:
            #         inpaint_mask = mask_gen(x)
            #         # inpaint_mask = inpaint_mask[:, 0, :, :].unsqueeze(dim=0)
            #         y = operator.forward(x, mask=inpaint_mask)
            #         cond_inpaint = inpaint_mask[:,0:1,:,:]
            #         cond_inpaint = resize(cond_inpaint,(64,64))
            # else: 
            #     # Forward measurement model (Ax + n)
            #     y = operator.forward(x)

            if args.train_with_noise:
                noiser.sigma = random.uniform(0, 0.1)
                if args.noise_sigma is not None:
                        noiser.sigma = args.noise_sigma
                y_n = noiser(y)
            else:
                y_n = y

            if args.use_denoiser:
                y_n = denoiser.predict((y_n/2 + 0.5).clamp(0,1))
                y_n = (y_n*2 - 1).clamp(-1,1)

            if "super_resolution" in args.task_config:
                y_n = resize(y_n,args.resolution,interpolation=InterpolationMode.BICUBIC,antialias=True)
                
            # Concatenate x and y_n along the batch dimension
            xy_n = torch.cat([x, y_n], dim=0)
            # # Encode both x and y_n together
            z_xy_n = diffusion_extractor.vae.encode(xy_n).latent_dist.sample() * diffusion_extractor.vae.config.scaling_factor
            # # Split the encoded tensor into z_x and z_y_n
            z_x, z_y_n = torch.chunk(z_xy_n, 2)
            imgs = z_x
            target = z_y_n
            
            sigma= (noiser.sigma/0.1)*999
            sigma_emb = embed_timestep(diffusion_extractor.unet, imgs, sigma)
            pred , _ = train_helpers.get_hyperfeats(diffusion_extractor, aggregation_network, imgs, eval_mode=True,is_latents=True,sigma_emb=sigma_emb,
                                                    kernel=None,cond_inpaint=cond_inpaint)
            loss = loss_spatial(aggregation_network, pred, target, sparse_loss, control_range,args=args)
            total_loss.append(loss.item())
            
            target = decode(target,diffusion_extractor.vae,all=True,normalized = False)
            pred = decode(pred,diffusion_extractor.vae,all=True,normalized = False)
            grid = train_helpers.log_grid(x[0:3], target[0:3], pred[0:3], control_range)
        break
    
    if split == "val":
        wandb.log({f'{split}/loss': loss,"validation": [wandb.Image(grid,caption=f"sigma = { noiser.sigma:.3f} grid imgs, target, pred") ]},step=step)

    elif split == "train":
        wandb.log({"training": [wandb.Image(grid,caption=f"sigma = { noiser.sigma:.3f} grid imgs, target, pred") ]},step=step)


def train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader,args,denoiser):
    # Load degredation configurations 

    dtype = torch.float32
    task_config = load_yaml(args.task_config)
    
    measure_config = task_config['measurement']
    
    if args.noise_sigma is not None:
        measure_config['noise']['sigma'] = args.noise_sigma
    if args.jpeg_qf is not None:
        measure_config['operator']['quality'] = args.jpeg_qf
    if args.sr_factor is not None and "resolution" in args.task_config:
        measure_config['operator']['scale_factor'] = args.sr_factor
    elif args.sr_factor is not None:
        print("given sr scale factor but the task is not super resolution. ignoring")
    if args.blur_sigma is not None:
        measure_config['operator']['intensity'] = args.blur_sigma
    
    operator = get_operator(device="cuda", **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])

    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
    else:
        inpaint_mask = None
        mask_gen = None

    doing_conditional_inpaint = measure_config['operator'] ['name'] == 'inpainting' and measure_config['mask_opt'].get("random",False)
    

    args.dps_operator = measure_config['operator']
    args.dps_noiser = measure_config['noise']
    
    if args.wandb_mode == "disabled":
        import lovely_tensors as lt
        lt.monkey_patch()
    
    run_id = wandb.run.name
    run_path = os.getcwd()
    wandb.run.log_code(
    "./",
    include_fn=lambda path: path.endswith("ulda_utils.py")
    )
    
    ckpts_base_path = os.path.join(run_path,"operator_ckpt",run_id)
    
    try:
        os.mkdir(ckpts_base_path)
    except(Exception):
        print("ckpts folders already exist, it might override ckpts in that folder")

    device = config.get("device", "cuda")
    run_name = config["id"]
    sparse_loss = config["dataset_args"]["sparse_loss"]
    control_range = config["dataset_args"]["control_range"]
    max_steps = config["max_steps"]
    max_epochs = math.ceil(max_steps / len(train_dataloader))

    aggregation_network = aggregation_network.to(device)
    train_end = False
    global_step = 0
    for epoch in range(max_epochs):
        for batch in tqdm(train_dataloader):
            with torch.no_grad():
                x = batch['pixel_values'].to("cuda",dtype=diffusion_extractor.model.dtype)

                if measure_config['operator'] ['name'] == 'inpainting':
                    inpaint_mask = mask_gen(x)
                    y = operator.forward(x, mask=inpaint_mask)
                else: 
                    y = operator.forward(x)

                if doing_conditional_inpaint:
                    cond_inpaint = inpaint_mask[:,0:1,:,:]
                    cond_inpaint = resize(cond_inpaint,(64,64)) #convert it to latent spatial dim
                else:
                    cond_inpaint = None


                if args.train_with_noise:
                    noiser.sigma = random.uniform(0, 0.1)
                    if args.noise_sigma is not None:
                        noiser.sigma = args.noise_sigma
                    y_n = noiser(y)
                else:
                    y_n = y

                if args.use_denoiser:
                    y_n = denoiser.predict((y_n/2 + 0.5).clamp(0,1))
                    y_n = (y_n*2 - 1).clamp(-1,1)

                if "super_resolution" in args.task_config:
                    if args.sr_legacy:
                        y_n = resize(y_n,args.resolution,interpolation=InterpolationMode.BICUBIC,antialias=True)
                    else:
                        y_n = operator.transpose(y_n)
                    
                xy_n = torch.cat([x, y_n], dim=0)
                # # Encode both x and y_n together
                z_xy_n = diffusion_extractor.vae.encode(xy_n).latent_dist.sample() * diffusion_extractor.vae.config.scaling_factor
                z_x, z_y_n = torch.chunk(z_xy_n, 2)
                
                    

                imgs = z_x
                target = z_y_n

                if config["clamp"] or args.clamp:
                    target = target.clamp(-4,4)

                sigma= (noiser.sigma/0.1)*999
                sigma_emb = embed_timestep(diffusion_extractor.unet, imgs, sigma)
                # kernel = operator.get_kernel()
          
            pred , latents = train_helpers.get_hyperfeats(diffusion_extractor, aggregation_network, imgs, eval_mode=False,is_latents=True,sigma_emb=sigma_emb,
                                                        kernel=None,cond_inpaint=cond_inpaint)
            
            loss = loss_spatial(aggregation_network, pred, target, sparse_loss, control_range,args=args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"train/loss": loss.item()}, step=global_step)
            
            if global_step > 0 and config["val_every_n_steps"] > 0 and global_step % config["val_every_n_steps"] == 0:
                with torch.no_grad():
                    pass
                    fig = train_helpers.log_aggregation_network(aggregation_network, config)
                    wandb.log({f"mixing_weights": fig}, step=global_step)
                    validate(config, diffusion_extractor, aggregation_network, train_dataloader, "train", global_step, run_name,operator,noiser,denoiser,inpaint_mask=inpaint_mask,mask_gen=mask_gen,doing_conditional_inpaint=doing_conditional_inpaint)
                    validate(config, diffusion_extractor, aggregation_network, val_dataloader, "val", global_step, run_name,operator,noiser,denoiser,inpaint_mask=inpaint_mask,mask_gen=mask_gen,doing_conditional_inpaint=doing_conditional_inpaint)
            if  global_step % config["save_model_steps"]== 0 and global_step > 0:
                pass
                train_helpers.save_model(config, aggregation_network, optimizer, global_step, run_name=run_name)
            global_step += 1
            if global_step == 5000:
                config["val_every_n_steps"] = 1000
                config["save_model_steps"] = 10000
            if global_step > max_steps:
                train_end = True
                break
        if train_end:
            break

def main(args):
    torch.set_default_dtype(torch.float32)
    config, diffusion_extractor, aggregation_network = train_helpers.load_models(args.config_path)
    args.config_file = config
    wandb.init(**config["wandb_kwargs"],mode=args.wandb_mode,config=vars(args),save_code=True,notes=args.notes)

    print("args = ", vars(args))
    print("--------------------------------")
    print("config = ",config)
    print(f"{aggregation_network.output_head_act=}")
    if args.start_from_ckpt is not None:
        aggregation_network.load_state_dict(torch.load(args.start_from_ckpt)['aggregation_network'], strict=False)
    
    wandb.run.name = f"{str(wandb.run.id)}_{wandb.run.name}"
    run_name = wandb.run.name
    
    optimizer = train_helpers.load_optimizer(config, diffusion_extractor, aggregation_network)
    '''
    dataset loading
    overall, you can change this code to whatever dataset you like,
    in the end, you just need to have a variable named: "dataset:
    which is a huggingface dataset object, where dataset["image"] returns a list of PIL images.
    '''
    if args.dataset_name == "coco":
        dataset = load_from_disk("data/coco/")
    elif args.dataset_name == "ffhq":
        dataset = load_from_disk("/home/ronraphaeli/my_datasets/ffhq512")
        dataset = dataset.train_test_split(train_size=1000,shuffle=False)
        val_dataset = dataset["train"] #this is actually the validation
        dataset = dataset["test"] #this is actually the train
    elif args.dataset_name == "LSDIR":
        dataset = load_from_disk("/home/ronraphaeli/my_datasets/LSDIR/")
        val_dataset = dataset["validation"]
        dataset = dataset["train"]
 
    
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.resolution, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.RandomVerticalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=.05, hue=.005),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) ,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    def preprocess_train(examples):
        examples["pixel_values"] = [train_transforms(image) for image in examples["image"]]
        return examples
    
    def preprocess_val(examples):
        examples["pixel_values"] = [val_transforms(image) for image in examples["image"]]
        return examples
    
    
    train_dataset = dataset.with_transform(preprocess_train)
    val_dataset = val_dataset.with_transform(preprocess_val)
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}
    def val_collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}
    
    
    
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=val_collate_fn,
        batch_size=3,
        num_workers=args.num_workers
    )

    config["dataset_args"]["sparse_loss"] = False
    config["dataset_args"]["control_range"] = (-1.0, 1.0)
    config["id"] = run_name
    config["dims"] = diffusion_extractor.dims


    import sys
    import os

    # Get the current working directory to ensure you know where you're starting
    current_dir = os.getcwd()
    print("Current directory:", current_dir)
    denoiser = None
    train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader,args,denoiser)

if __name__ == "__main__":
    
    args = parse_args()
    if args.wandb_mode == "disabled":
        import lovely_tensors
        lovely_tensors.monkey_patch()
    main(args)