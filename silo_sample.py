#usual imports
import os
import re
import warnings
import json
import argparse
import time
from PIL import Image
import lovely_tensors as lt
lt.monkey_patch()


#general ml imports
import datasets
from datasets import load_from_disk
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image , to_tensor  , resize
from torchvision.transforms import InterpolationMode
from kornia.color import rgb_to_ycbcr
from diffusers import DDPMScheduler

#SILO imports
from util.img_utils import  mask_generator
from guided_diffusion.measurements import get_noise, get_operator
from silo_utils import DegradationModel, make_grid, set_seeds, load_yaml , StableDiffusionPipeline , PosteriorSampling, add_text_to_grid_sep, ensure_directory_structure, parse_list_from_string, is_debugging, VaeWrapper, get_command_line

#remove warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='torch')
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers')
warnings.filterwarnings("ignore", category=FutureWarning, module='kornia')
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub')
warnings.simplefilter(action='ignore', category=FutureWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None ,
        # default="stablediffusionapi/realistic-vision-v51" ,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ffhq",
        choices=["ffhq","coco"],
        help=("The name of the Dataset"),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A high quality photo of a face",
        help="A prompt that is used to guide the restoration",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="A negative prompt that is used to guide the restoration",
    )
    parser.add_argument(
        "--seed", type=int, default=1000, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=("The resolution images"),
    )
    parser.add_argument(
        "--start_from_ckpt",
        type=str,
        default=None,
        help=("the checkpoint of the learned latent operator"),
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default="configs/gaussian_deblur_config.yaml",
        help=("the config of the degredation task"),
    )
    parser.add_argument(
        "--noise_sigma",
        type=float,
        default=None,
        help="manual way to control the noise sigma of the measurment creation",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=1.0,
        help="cfg",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="the scale of the gradient step. called eta in SILO paper",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=999,
        help="number of inference step",
    )
    parser.add_argument(
        "--sr_factor",
        type=int,
        default=None,
        help="manual way to force a certain sr factor on super resolution task",
    )
    parser.add_argument(
        "--idx_image",
        type=int,
        default=1,
        help="the idx of the photo from the dataset",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        required=False,
        help=("pass in order to create it for a specific image with an abs path"),
    )
    parser.add_argument(
        "--explicit_name",
        type=str,
        default=None,
        required=False,
        help=("pass in order to create it for a specific image with an abs path"),
    )
    parser.add_argument(
        "--sigma_condition",
        default=False,
        action="store_true",
        help="to pass if the network is conditioned on the noise sigma in the measurment. in SILO this is the case",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="if you want that the sampling will save a note on this run in its log file",
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        default="./sample_results",
        help="the folder for the results of the algorithm",
    )
    parser.add_argument(
        "--clamp",
        default=False,
        action="store_true",
        help="pass in order to perform w.clamp(-4,4) as described in SILO in Algorithm 1. this help the restorations to be a bit more 'crisp' since it cleans out encoding problems",
    )
    parser.add_argument(
        "--save_mode",
        type=str,
        default="grid",
        choices=["grid", "image", "both"],
        help=(
            "to save the grid or just the image"
    ))
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="the start idx of the photo from the ",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=1,
        help="the start idx of the photo from the ",
    )
    parser.add_argument(
        "--disable_bar",
        default=False,
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--list_to_do",
        type=str,
        default=None,
        help="list of the experiments idx to do",
    )
    parser.add_argument(
        "--hq_captions",
        default=False,
        action="store_true",
        help="in case you want to use the hq prompts as described in SILO",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rg",
        choices=["rg", "cnn", "preserve_LatentDegradationNetwork"],
        help=(
            "if the operator is of readout guidance or a CNN"
    ))
    parser.add_argument(
        "--diff_model",
        type=str,
        default="rv",
        choices=["sd1.5","rv",None],
        help=("use this instead of pretrained model name or path")
        )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16"],
        help=(
            "to save the grid or just the image"
    ))
    parser.add_argument(
        "--print_vars",
        default=False,
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--cfg_fix",
        default=False,
        action="store_true",
        help="if you pass cfg>1 , you should enable it, but it is not a must. it stabilize cfg sampling",
    )
    parser.add_argument(
        "--deterministic",
        default=False,
        action="store_true",
        help="to create the sampling pixel perfect deterministic (fyi, since some torch algorithms are stochacstic, passing only the same seed does not always creates exactly the same output)",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f"Unknown arguments: {' '.join(unknown)}")
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.list_to_do is not None:
        args.list_to_do = parse_list_from_string(args.list_to_do)
    return args


def main():
    if is_debugging():
        print("Running under a debugger")
        import sys
        sys.argv = [
            "silo_sample.py",
            "--prompt", "A high quality photo of a face",
            "--start_from_ckpt", "trained_ckpt/RG/ip_rg_coco_operator.pt",
            "--task_config", "configs/center_inpainting_config.yaml",
            "--scale", "0.5",
            "--idx_image", "24",
            "--model", "rg",
            "--cfg", "1",
            "--sigma_condition", "--clamp",
            "--steps", "50"
        ]
    else:
        print("Running normally")
    
    args = parse_args()
    ensure_directory_structure(args.folder_path,save_mode=args.save_mode)
    command_line_exec = get_command_line()
    device = torch.device("cuda")
    global_dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    torch.set_default_dtype(global_dtype)
    run_path = os.getcwd()
    
    #load the task
    task_config = load_yaml(args.task_config)
    measure_config = task_config['measurement']
    
    operator =  get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    
    args.dps_operator = measure_config['operator']
    args.dps_noiser = measure_config['noise']

    
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
    doing_conditional_inpaint = measure_config['operator'] ['name'] == 'inpainting' and measure_config['mask_opt'].get("random",False)
    

    '''
    dataset loading
    overall, you can change this code to whatever dataset you like,
    in the end, you just need to have a variable named: "dataset:
    which is a huggingface dataset object, where dataset["image"] returns a list of PIL images.
    '''
    if args.image_path is not None:
        args.image_path = os.path.join(run_path,args.image_path)
        img = Image.open(args.image_path)
        dataset = {'image':[img]}
        dataset = datasets.Dataset.from_dict(dataset)
        args.idx_image = 0
    elif args.dataset_name == "coco":
        dataset = load_from_disk("./data/coco/")
    elif args.dataset_name == "ffhq":
        dataset = load_from_disk("./data/ffhq/")

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) ,
            transforms.ToTensor(),
        ]
    )
    
    def preprocess_train(examples):
        examples["pixel_values"] = [train_transforms(image) for image in examples["image"]]
        return examples
    
    dataset = dataset.with_transform(preprocess_train)
    normalizer = transforms.Normalize([0.5], [0.5])
        
    #load metrics
    psnr_calc = PeakSignalNoiseRatio(data_range=(-1,1)).to("cuda")
    psnr_calc_0_to_1 = PeakSignalNoiseRatio(data_range=(0,1)).to("cuda")
    lpips_alex_calc = LearnedPerceptualImagePatchSimilarity(net_type='alex').to("cuda")
    lpips_vgg_calc = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to("cuda")
    ms_ssim_calc = MultiScaleStructuralSimilarityIndexMeasure(data_range=(-1,1)).to("cuda")

    vae_wrapper = VaeWrapper(pretrained_model_name_or_path=args.pretrained_model_name_or_path,type=args.diff_model)
    vae_type = vae_wrapper.get_vae_type()
    num_latent_channels = vae_wrapper.get_num_latent_channels()
    args.pretrained_model_name_or_path = vae_wrapper.get_pretrained_model_name_or_path()

    #load pipe
    if vae_type in ("sd1.5", "rv"):
        pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=global_dtype , requires_safety_checker=False,safety_checker=None,
                                                    sigma_condition=args.sigma_condition,sampling_args=args,
                                                    cond_inpaint=doing_conditional_inpaint 
                                                    ).to("cuda")
        pipe.scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=global_dtype,subfolder="scheduler")
    else:
        raise NotImplemented("choose pipe")
    
    if args.model == "rg":
        pipe.dps_sampler = PosteriorSampling(operator=None, noiser=noiser,scale=args.scale)
        pipe.aggregation_network.load_state_dict(torch.load(args.start_from_ckpt,weights_only=True)['aggregation_network'], strict=True)
        pipe.aggregation_network = pipe.aggregation_network.to("cuda")
        pipe.aggregation_network.eval()
    elif args.model == "cnn":
        model = DegradationModel(num_latent_channels,layer_norm=True,sigma_condition=args.sigma_condition)
        model.load_state_dict(torch.load(args.start_from_ckpt),strict=True)
        model = model.to(device)
        model.eval()
        pipe.dps_sampler = PosteriorSampling(operator=model, noiser=noiser,scale=args.scale)
    elif args.model == "preserve_LatentDegradationNetwork":
        from preserve_LatentDegradationNetwork import LatentDegradationNetwork
        model = LatentDegradationNetwork(
            in_channels=4,
            out_channels=4,
            base_channels=48,  
            num_blocks=3,
            max_noise=0.1
            )
        model.load_state_dict(torch.load(args.start_from_ckpt),strict=True)
        model = model.to(device)
        model.eval()
        pipe.dps_sampler = PosteriorSampling(operator=model, noiser=noiser,scale=args.scale)
    else:
        raise NotImplemented("choose a model that exists")
        
    if args.disable_bar:
        pipe.set_progress_bar_config(disable=True)

    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"running on gpu: {gpu_name}")
    if args.print_vars:
        print("sampling with args:")
        print(vars(args))
    
    #the logic here is to handle many cases of sampling. e.g, we have a start and end arguments, or a list of indexs or a single image
    if args.list_to_do is not None:
        idx_to_work_on = args.list_to_do
        batch_sampling_mode = True
    elif (args.start != 0) or (args.end != 1):
        idx_to_work_on = range(args.start,args.end)
        batch_sampling_mode = True
    else:
        idx_to_work_on = [args.idx_image]
        batch_sampling_mode = False

    #the actual for loop of the reconstruction
    for real_idx in idx_to_work_on: #each iteration in this for loop is of a different reconstruction
        if batch_sampling_mode:
            args.idx_image = real_idx
            args.explicit_name = real_idx
            print(f"starting to sample {real_idx}")
            if real_idx != idx_to_work_on[0]:
                del x_og
            
        set_seeds(args.seed + doing_conditional_inpaint*real_idx,deterministic=args.deterministic) #if we do conditional inpaint than we want to create a different mask for each example in batch, so we must give each a diff seed

        #first, we create y and y_n
        #in the code, x is an image, y is a degraded image and y_n is y plus noise
        with torch.no_grad():
            if args.idx_image is not None:

                x_og = dataset[args.idx_image]['pixel_values'][0:3].unsqueeze(0).to(device)
                x = normalizer(x_og)

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

                # y_n = y + noise
                y_n = noiser(y) #since the images here are [-1,1] images, the true std is half compared to the config

                #in case the task is SR, we need to scale back to args.resolution
                if "super_resolution" in args.task_config:
                    y_n = resize(y_n,args.resolution,interpolation=InterpolationMode.BICUBIC,antialias=True)

                w = vae_wrapper.encode(y_n)
                
                    

        if args.hq_captions: #if you have a dataset of captions
            args.prompt = hq_captions[args.idx_image]['caption']
        if args.clamp:
            w = w.clamp(-4,4)

        #the reconstruction pipeline
        start_time = time.time()
        set_seeds(args.seed,deterministic=args.deterministic)
        image = pipe(args.prompt, negative_prompt=args.negative_prompt, #prompt and negative prompt
                      guidance_scale=args.cfg, height=args.resolution, width=args.resolution,num_inference_steps=args.steps, #regular LDM args
                      do_dps=True, w=w.clone(),
                      sampling_args=args, cond_inpaint=cond_inpaint
                    ).images[0]
        end_time = time.time() 

        #preliminary stage to calculate metrics
        image_tensor = to_tensor(image).unsqueeze(0).to(device).requires_grad_(False)
        image_tensor_normed = normalizer(image_tensor)
        if measure_config['operator'] ['name'] == 'inpainting':
            image_tensor_y = operator.forward(image_tensor_normed, mask=inpaint_mask)
        else: 
            image_tensor_y = operator.forward(image_tensor_normed)
        ycbcr_image = rgb_to_ycbcr(image_tensor)
        ycbcr_x_og = rgb_to_ycbcr(x_og[0:1])

        #calculate metrics
        psnr = psnr_calc(image_tensor_normed,x[0:1]).cpu().item()
        ms_ssim = ms_ssim_calc(image_tensor_normed,x[0:1]).cpu().item()
        Y_psnr = psnr_calc_0_to_1(ycbcr_image[:,0,:,:],ycbcr_x_og[:,0,:,:]).cpu().item()
        lpips_alex = lpips_alex_calc(image_tensor_normed,x[0:1]).cpu().item()
        lpips_vgg = lpips_vgg_calc(image_tensor_normed,x[0:1]).cpu().item()
        
        if "super_resolution" in args.task_config:
            y = resize(y,args.resolution,interpolation=InterpolationMode.BICUBIC,antialias=True)
            image_tensor_y = resize(image_tensor_y,args.resolution,interpolation=InterpolationMode.BICUBIC,antialias=True)

        consistency_psnr = psnr_calc(image_tensor_y,y[0:1]).cpu().item()
        Y_consistency_psnr = psnr_calc_0_to_1(rgb_to_ycbcr(((image_tensor_y/2)+0.5).clamp(0,1))[:,0,:,:],rgb_to_ycbcr(((y[0:1]/2)+0.5).clamp(0,1))[:,0,:,:]).cpu().item()
        
        print(f"x_og to image {lpips_alex = :.3f} {lpips_vgg = :.3f}")
        print(f"x_og to image {psnr = :.2f}  {consistency_psnr = :.2f}")
        print(f"Y of YCbCr {Y_psnr = :.2f}  {Y_consistency_psnr = :.2f}")


        #saving code
        files = os.listdir(args.folder_path)
        numeric_png_files = [file for file in files if re.match(r'^\d+\.png$', file)]
        if numeric_png_files == []:
            max_number = -1
        else:
            # Extract numbers from filenames and find the maximum
            numbers = [int(file.split(".")[0]) for file in numeric_png_files]
            max_number = max(numbers)

        file_name = args.explicit_name if args.explicit_name is not None else f"{max_number + 1}"

        with torch.no_grad():
            grid = make_grid(to_pil_image(x_og[0].float()).resize((512,512))
                            ,to_pil_image((y_n[0].float()/2 + 0.5).clamp(0,1).cpu()).resize((512,512))
                            ,image.resize((512,512))
                            ,to_pil_image((image_tensor_y[0].float()/2+0.5).clamp(0,1).cpu()).resize((512,512))
                            )
            grid = add_text_to_grid_sep(grid,["x" , "A(x) + n" , "output", "A(output)"], f"{file_name}: noise sigma = {noiser.sigma} ::: {psnr = :.2f} {lpips_alex = :.3f} {lpips_vgg = :.3f} {consistency_psnr = :.3f}", font_size=20,im_size=args.resolution)
        if args.save_mode == "grid":
            grid.save(os.path.join(args.folder_path,f"{file_name}.png"))
        elif args.save_mode == "image":
            image.save(os.path.join(args.folder_path,f"{file_name}.png"))
        elif args.save_mode == "both":
            grid.save(os.path.join(args.folder_path,"grids",f"{file_name}.png"))
            image.save(os.path.join(args.folder_path,"images",f"{file_name}.png"))
        else:
            raise NotImplementedError

        notes = args.notes[:]
        log_data = {
            'args': {k: v for k, v in vars(args).items() if k != 'notes'},
            'metrics': {
                'psnr': psnr,
                'consistency_psnr': consistency_psnr, 
                'lpips_alex': lpips_alex,
                'lpips_vgg': lpips_vgg,
                'Y_psnr': Y_psnr, 
                'Y_consistency_psnr': Y_consistency_psnr, 
                'ms-ssim':ms_ssim,
                'sampling_time': round(end_time - start_time,2)
            },
            'notes': notes if notes is not None else '',
            "gpu_name": gpu_name,
            "run_command": command_line_exec
        } 
        with open(os.path.join(args.folder_path,"logs",f"{file_name}.json"), 'w') as f:
                json.dump(log_data, f, indent=4)

        print(f"saved as {file_name}.png")

    
if __name__ == "__main__":
    main()

        