import torchvision.transforms as transforms
import os
import random
import wandb
import pandas as pd
import argparse
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image  , resize
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import sys
sys.path.append("../")
from diffusers import (AutoencoderKL, UNet2DConditionModel)
from guided_diffusion.measurements import get_noise, get_operator
from silo_utils import set_seeds, load_yaml, DegradationModel , embed_timestep, VaeWrapper
from datasets import load_from_disk
from util.img_utils import  mask_generator
torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        # "botp/stable-diffusion-v1-5" or "stablediffusionapi/realistic-vision-v51" ,
        default="stablediffusionapi/realistic-vision-v51",
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
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
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
        "--f",
        type=str,
        default="",
        required=False,
        help=("ignore"),
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
        "--sr_factor",
        type=int,
        default=None,
        help="noise for dps noiser",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--train_with_noise",
        default=False,
        action="store_true",
        help="if to train with noise",
    )
    parser.add_argument(
        "--l1_loss",
        default=False,
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--sigma_condition",
        action="store_true",
        default=False,
        help="the idx of the photo from the dataset",
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
        "--model_type",
        type=str,
        default="silo_cnn",
        choices=["silo_cnn", "preserve_LatentDegradationNetwork"],
        help=(
            "the model type to train"
    ))
    parser.add_argument(
        "--opt_type",
        type=str,
        default="adam",
        choices=["adam", "adamw"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--add_adv_noise",
        default=False,
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--adv_noise_sigma",
        type=float,
        default=0.001,
        help="Epsilon value for the Adam optimizer",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Need either a dataset name or a training folder.")

    return args

import sys
import traceback
import os

def is_debugging():
    if sys.gettrace() is not None:
        return True
    for frame in traceback.extract_stack():
        if frame.filename and 'pydevd' in frame.filename:
            return True
    return any('PYCHARM' in key or 'PYDEVD' in key for key in os.environ)

def main():
    if is_debugging():
        print("Running under a debugger")
        import sys
        sys.argv = [
            "script_name.py",  # This would be replaced by the actual script name
            "--task_config", "",
            "--noise_sigma", "0.02",
            "--learning_rate=1e-3",
            "--wandb_mode=disabled",
            "--sigma_condition",
            "--train_batch_size=2",
        ]
    else:
        print("Running normally")
    args = parse_args()
    
    save_every_steps = 300 # save the model ckpt every save_every_steps global steps
    num_train_epochs= 100
    save_photos = 250 # tries to save the training and validation photos every save_photos steps. will only enter if a validation happend in the same glob step
    do_validtion_steps = 25 # do validation every do_validtion_steps global steps
    wandb_project_name = "silo_iccv"
    
    device = torch.device("cuda")
    
    # Load degredation configurations 
    task_config = load_yaml((args.task_config))
    
    measure_config = task_config['measurement']
    
    if args.jpeg_qf is not None:
        measure_config['operator']['quality'] = args.jpeg_qf
    if args.noise_sigma is not None:
        measure_config['noise']['sigma'] = args.noise_sigma
    if args.sr_factor is not None and "resolution" in args.task_config:
        measure_config['operator']['scale_factor'] = args.sr_factor
    elif args.sr_factor is not None:
        print("given sr scale factor but the task is not super resolution. ignoring")
    
    if isinstance(measure_config['operator'],list):
        operator = get_operator(device="cuda", **{"name": measure_config['operator']})
    else:
        operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])

    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        inpaint_mask = mask_gen(None)
        inpaint_mask = inpaint_mask[:, 0, :, :].unsqueeze(dim=0)

    args.dps_operator = measure_config['operator']
    args.dps_noiser = measure_config['noise']
    
    wandb.init(project=wandb_project_name,
              config=vars(args),
              save_code=True,
              notes=args.notes,
              mode=args.wandb_mode,
              group=args.model_type,)
    
    if args.wandb_mode == "disabled":
        import lovely_tensors as lt
        lt.monkey_patch()
    run_id = wandb.run.name
    run_path = os.getcwd()

    os.makedirs(os.path.join(run_path,"cnn_ckpt"), exist_ok=True)
    ckpts_base_path = os.path.join(run_path,"cnn_ckpt",run_id)
    
    try:
        os.mkdir(ckpts_base_path)
    except(Exception):
        print("ckpts folders already exist, it might override ckpts in that folder")
    
    '''
    dataset loading
    overall, you can change this code to whatever dataset you like,
    in the end, you just need to have a variable named: "dataset:
    which is a huggingface dataset object, where dataset["image"] returns a list of PIL images.
    '''
    if args.dataset_name == "coco":
        # dataset = load_from_disk("data/coco/")
        dataset = load_from_disk("/home/ronraphaeli/my_datasets/COCO/")
    elif args.dataset_name == "ffhq":
        dataset = load_from_disk(("/home/ronraphaeli/my_datasets/ffhq512/"))
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
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers
    )
    
    #load vae
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to("cuda")
    vae.requires_grad_(False)
    vae_type = "sd1.5"
    num_latent_channels = 4
    vae_wrapper = VaeWrapper(vae,type=vae_type)
    if args.sigma_condition:
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to("cuda")
        unet.requires_grad_(False)
    
    set_seeds(args.seed)
    
    print(f"model type: {args.model_type}")
    if args.model_type == "silo_cnn":
        model = DegradationModel(num_latent_channels,layer_norm=True,sigma_condition=args.sigma_condition)
        def model_fn(z, sigma_emb, **kwargs):
            w_hat = model(z, sigma_emb=sigma_emb)
            return w_hat
    elif args.model_type == "preserve_LatentDegradationNetwork":
        from preserve_LatentDegradationNetwork import LatentDegradationNetwork
        model = LatentDegradationNetwork(
        in_channels=4,
        out_channels=4,
        base_channels=48,  
        num_blocks=3,
        max_noise=0.1
        )
        def model_fn(z, sigma, **kwargs):
            sigma = torch.tensor([sigma] * z.shape[0]).to(z.device)
            return model(x=z, noise_level=sigma)
    else:
        raise NotImplementedError("model type not implemented")
    
    if args.start_from_ckpt is not None:
        model.load_state_dict(torch.load(args.start_from_ckpt,weights_only=True),strict=True)
        print(f"init the weights of the model from {args.start_from_ckpt}")
        
    model = model.to("cuda")
    model.train()
    if args.opt_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.opt_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        raise NotImplementedError("can choose from adam or adamw")

    global_step = 0
    set_seeds(args.seed)
    for epoch in range(0, num_train_epochs):
        for step, batch in tqdm(enumerate(train_dataloader)):

            with torch.no_grad():
                x = batch['pixel_values'].to("cuda")
                if measure_config['operator'] ['name'] == 'inpainting':
                    y = operator.forward(x, mask=inpaint_mask)
                else: 
                    y = operator.forward(x)
                if args.train_with_noise:
                    noiser.sigma = random.uniform(0, 0.04)
                    if args.noise_sigma is not None:
                        noiser.sigma = args.noise_sigma
                    y_n = noiser(y)
                else:
                    y_n = y

                if "super_resolution" in args.task_config or True:
                    y_n = resize(y_n,args.resolution,interpolation=InterpolationMode.BICUBIC,antialias=True)
                    y = resize(y,args.resolution,interpolation=InterpolationMode.BICUBIC,antialias=True)
                
               
                target = y_n

                xy_n = torch.cat([x, target], dim=0)
                z_xy_n = vae_wrapper.encode(xy_n)
                imgs, target = torch.chunk(z_xy_n, 2) 

                if args.clamp:
                    target = target.clamp(-4,4)
            
            if args.sigma_condition:
                sigma_emb = embed_timestep(unet, imgs, (noiser.sigma/0.1)*999)
            else:
                sigma_emb = None
            if args.add_adv_noise:
                imgs = imgs + torch.randn_like(imgs) * args.adv_noise_sigma
                
            model_output = model_fn(z = imgs, sigma_emb=sigma_emb, sigma = noiser.sigma)

            if args.l1_loss:
                loss = F.l1_loss(model_output, target) 
            else:
                loss = F.mse_loss(model_output, target) 
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step == 5000:
                save_photos = 2000
            if global_step == 10000:
                save_every_steps = 4000
            optimizer.zero_grad()
            
            #validation
            with torch.no_grad():
                wandb.log(
                {
                    "epoch": epoch,
                    "train_loss":loss.cpu().detach().item()
                },
                step=global_step, )

                if global_step % do_validtion_steps == 0 :
                    model.eval()

                    for _ , batch_val in enumerate(val_dataloader):
                        x_val = batch_val['pixel_values'].to("cuda")
                        break

                    if measure_config['operator'] ['name'] == 'inpainting':
                        y_val = operator.forward(x_val, mask=inpaint_mask)
                    else: 
                        y_val = operator.forward(x_val)

                    if "super_resolution" in args.task_config or True:
                        y_val = resize(y_val,args.resolution,interpolation=InterpolationMode.BICUBIC,antialias=True)
                        
                    if args.train_with_noise:
                        noiser.sigma = random.uniform(0, 0.04)
                        if args.noise_sigma is not None:
                            noiser.sigma = args.noise_sigma
                        y_n_val = noiser(y_val)
                    else:
                        y_n_val = y_val

                    if args.sigma_condition:
                        sigma_emb = embed_timestep(unet, imgs, (noiser.sigma/0.1)*999)
                    else:
                        sigma_emb = None

                    
                    target_val = y_n_val

                    xy_n_val = torch.cat([x_val, target_val], dim=0)
                    z_xy_n_val = vae_wrapper.encode(xy_n_val)
                    img_val, target_val = torch.chunk(z_xy_n_val, 2) 

                    if args.clamp:
                        target_val = target_val.clamp(-4,4)
                    model_output_val = model_fn(z = img_val, sigma_emb=sigma_emb, sigma = noiser.sigma)

                    if args.l1_loss:
                        loss_val = F.l1_loss(model_output_val, target_val)
                    else:
                        loss_val = F.mse_loss(model_output_val, target_val)
                    model.train()
                    
                    wandb.log(
                                {
                                    "epoch": epoch,
                                    "val_loss":loss_val.cpu().detach().item()
                                },
                                step=global_step,
                                )
                
                if global_step % save_photos == 0 and global_step % do_validtion_steps == 0:
                    wandb.log({
                            "training": [wandb.Image(to_pil_image((x[0]/2 + 0.5).clamp(0,1)).resize((256, 256)),
                                                caption=f"x"),
                                         wandb.Image(to_pil_image((y_n[0]/2 + 0.5).clamp(0,1)).resize((256, 256)),
                                                caption=f"y_n sigma = {noiser.sigma:.3f}"),
                                         wandb.Image(vae_wrapper.decode(model_output[0:1],return_as="pil image").resize((256, 256)),
                                                caption=f"decoded model output") ],
                            "validation": [wandb.Image(to_pil_image((x_val[0]/2 + 0.5).clamp(0,1)).resize((256, 256)),
                                                caption=f"x"),
                                         wandb.Image(to_pil_image((y_n_val[0]/2 + 0.5).clamp(0,1)).resize((256, 256)),
                                                caption=f"y_n sigma = {noiser.sigma:.3f}"),
                                         wandb.Image(vae_wrapper.decode(model_output_val[0:1],return_as="pil image").resize((256, 256)),
                                                caption=f"decoded model output")]
                                    },step=global_step, )
                    
                if global_step % save_every_steps == 0 :
                    file_name = f"model_weights_step_{global_step}.pth"
                    final_path = os.path.join(ckpts_base_path,file_name)
                    torch.save(model.state_dict(), final_path)
                    print(f"saved a ckpt! {global_step = }  {epoch = }  {step = }") 
                    
            

if __name__ == "__main__":
    main()