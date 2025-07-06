import yaml
import os
import random
from typing import Optional, Union
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import sys
import traceback
import os

import torch
import torch.nn as nn
import einops
from diffusers import (AutoencoderKL)

# from DPS
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

from abc import ABC, abstractmethod
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, w, w_hat,**kwargs):
        difference = w - w_hat
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass

class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self._adjust_scale = 1.0

    def conditioning(self, x_prev, x_t, x_0_hat, w,w_hat, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat,
                                              w=w, w_hat=w_hat,
                                              **kwargs)
        x_t -= norm_grad.clamp(-1,1) * self.scale * self._adjust_scale
        return x_t, norm

#misc utils for SILO code
def add_text_to_grid_sep(grid, texts_above, text_below, font_path=None, font_size=20,im_size=512):
    """
    Adds text above each image in a grid of concatenated images and one line of text below the entire grid.

    Args:
    - grid (PIL.Image): The grid image containing concatenated images.
    - texts_above (list of str): The list of texts to add above each image. Each image has its own text.
    - text_below (str): The text to add below the entire grid.
    - font_path (str, optional): Path to the font file to use. If None, default font is used.
    - font_size (int): The size of the font.

    Returns:
    - Image: The resulting image with text added.
    """
    # Ensure the grid image has the correct height (512 pixels)
    width, height = grid.size
    if height != im_size:
        raise ValueError("Grid image height must be 512 pixels.")

    # Calculate the number of images in the grid
    num_images = width // im_size
    if width % im_size != 0:
        raise ValueError("Grid width must be a multiple of 512 pixels.")

    # Ensure the texts_above list has the correct number of texts
    if len(texts_above) != num_images:
        raise ValueError(f"The number of texts ({len(texts_above)}) must match the number of images ({num_images}).")

    # Create a new image to hold the grid and the text
    new_height = height + 60  # Add space for text above and below
    new_grid = Image.new('RGB', (width, new_height), (255, 255, 255))

    # Paste the original grid into the new image
    new_grid.paste(grid, (0, 30))  # Offset by 30 pixels to leave space for text above

    # Create a drawing context
    draw = ImageDraw.Draw(new_grid)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    # Add text above each image
    for i in range(num_images):
        image_x = i * im_size
        
        # Text above the image
        text_above = texts_above[i]
        text_bbox_above = draw.textbbox((0, 0), text_above, font=font)
        text_width_above, text_height_above = text_bbox_above[2] - text_bbox_above[0], text_bbox_above[3] - text_bbox_above[1]
        text_x_above = image_x + (im_size - text_width_above) // 2
        draw.text((text_x_above, 10), text_above, font=font, fill="black")  # 10 pixels down from the top

    # Add the single text below the entire grid
    text_bbox_below = draw.textbbox((0, 0), text_below, font=font)
    text_width_below, text_height_below = text_bbox_below[2] - text_bbox_below[0], text_bbox_below[3] - text_bbox_below[1]
    text_x_below = (width - text_width_below) // 2
    draw.text((text_x_below, height + 40 - text_height_below // 2), text_below, font=font, fill="black")  # 40 pixels down from the original image height

    return new_grid

def check_folder_for_1000_pngs(folder_path):
    """
    Checks if a folder contains exactly 1000 PNG files named from 0.png to 999.png
    
    Args:
        folder_path (str): Path to the folder to check
        
    Returns:
        bool: True if validation passes, False otherwise
        list: List of missing files (if any)
    """
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        return False, ["Folder does not exist"]
    
    # Get all PNG files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    # Check if there are exactly 1000 PNG files
    if len(files) != 1000:
        return False, [f"Found {len(files)} PNG files instead of 1000"]
    
    # Check if all expected files exist
    expected_files = [f"{i}.png" for i in range(1000)]
    missing_files = [f for f in expected_files if f not in files]
    
    # Check if there are any extra files
    extra_files = [f for f in files if f not in expected_files]
    
    if missing_files or extra_files:
        issues = []
        if missing_files:
            issues.append(f"Missing files: {missing_files[:10]}..." if len(missing_files) > 10 else f"Missing files: {missing_files}")
        if extra_files:
            issues.append(f"Extra files: {extra_files[:10]}..." if len(extra_files) > 10 else f"Extra files: {extra_files}")
        return False, issues
    
    return True, []

def extract_image_from_grid(grid_image, index, im_size=512):
    """
    Extracts a single image from a grid of images based on its index.
    
    Args:
    - grid_image (PIL.Image): The grid image containing concatenated images.
    - index (int): The index of the image to extract (0 is the first, 3 is the last in a grid of 4).
    - im_size (int): The size of each image in the grid (assumes square images).
    
    Returns:
    - Image: The extracted image.
    """
    # Get grid dimensions
    width, height = grid_image.size
    
    # Calculate the number of images in the grid
    num_images = width // im_size
    
    # Check if the index is valid
    if index < 0 or index >= num_images:
        raise ValueError(f"Index {index} is out of bounds. Grid contains {num_images} images.")
    
    # Check if the grid image includes text areas (extra height)
    is_annotated = height > im_size
    
    # Determine the vertical offset for extracting the image
    # If the grid has text annotations (from add_text_to_grid_sep), the images start at y=30
    y_offset = 30 if is_annotated else 0
    
    # Calculate the position of the image in the grid
    left = index * im_size
    top = y_offset
    right = left + im_size
    bottom = top + im_size
    
    # Crop and return the image
    return grid_image.crop((left, top, right, bottom))

def plot_and_save(data, save_path,title):
    """
    Plots the data and saves the plot to the specified path.

    Parameters:
        data (list of tuples): A list where each element is a tuple (timestep, norm, scale).
        save_path (str): The file path where the plot will be saved.
    """
    # Extract the components of the tuples
    timesteps = [t[0] for t in data]
    norms = [t[1] for t in data]
    scales = [t[2] * 30 for t in data]

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, norms, label='Norm')
    plt.plot(timesteps, scales, label='Scale')

    plt.xlabel('Timestep')
    plt.ylabel('Values')
    plt.title(f"{title}")
    plt.legend()
    plt.grid(True)

    # Reverse the x-axis
    plt.gca().invert_xaxis()

    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()

def ensure_directory_structure(folder_path,save_mode="image"):

    # Check if the folder path exists
    if not os.path.exists(folder_path):
        # Create the main folder if it doesn't exist
        os.makedirs(folder_path)
        print(f"Created main folder: {folder_path}")

    # Define the subfolders
    subfolders = ['logs', 'plots']
    if save_mode == "both":
        subfolders.append("images")
        subfolders.append("grids")

    # Check and create each subfolder if it doesn't exist
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path, exist_ok=True)
            print(f"Created subfolder: {subfolder_path}")

def parse_list_from_string(list_string):
    # Remove brackets if present, strip spaces, then split by commas
    list_string = list_string.strip("[]")
    items = list_string.split(",")
    
    # Convert the items to integers
    return [int(item.strip()) for item in items if item.strip()]

def is_debugging():
    if sys.gettrace() is not None:
        return True
    for frame in traceback.extract_stack():
        if frame.filename and 'pydevd' in frame.filename:
            return True
    return any('PYCHARM' in key or 'PYDEVD' in key for key in os.environ)

def make_grid(*images, rows=1):
    # Ensure all elements in images are PIL Image objects
    assert all(isinstance(image, Image.Image) for image in images), "All inputs must be PIL Image objects"
    
    # Ensure there is at least one image
    assert len(images) > 0, "There must be at least one image"
    
    # Ensure all images are the same size
    widths, heights = zip(*(image.size for image in images))
    assert all(width == widths[0] for width in widths), "All images must have the same width"
    assert all(height == heights[0] for height in heights), "All images must have the same height"
    
    # Ensure rows is valid
    assert rows > 0, "Rows must be greater than 0"
    assert rows <= len(images), "Rows cannot be greater than number of images"

    # Calculate images per row (ceiling division to handle uneven divisions)
    images_per_row = -(-len(images) // rows)  # Equivalent to math.ceil(len(images) / rows)
    
    # Calculate grid dimensions
    single_width = widths[0]
    single_height = heights[0]
    total_width = single_width * images_per_row
    total_height = single_height * rows
    
    # Create the new image
    grid_image = Image.new('RGB', (total_width, total_height))
    
    # Paste the images into the grid
    for idx, image in enumerate(images):
        row = idx // images_per_row
        col = idx % images_per_row
        x_offset = col * single_width
        y_offset = row * single_height
        grid_image.paste(image, (x_offset, y_offset))
    
    return grid_image

def set_seeds(seed,deterministic=False):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        np.random.seed(seed)
        random.seed(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            torch.use_deterministic_algorithms(True, warn_only=False)


import psutil
import os

def get_command_line():
    args = psutil.Process(os.getpid()).cmdline()
    reconstructed = []
    
    for arg in args:
        # Handle --flag=value cases
        if '=' in arg:
            flag, value = arg.split('=', 1)
            # If original value has spaces or quotes, wrap in quotes without escaping
            if ' ' in value or '"' in value:
                # Replace any existing double quotes with single quotes
                value = value.replace('"', "'")
                reconstructed.append(f'{flag}="{value}"')
            else:
                reconstructed.append(f'{flag}={value}')
        else:
            # For regular arguments
            if ' ' in arg or '"' in arg:
                # Replace any existing double quotes with single quotes
                arg = arg.replace('"', "'")
                reconstructed.append(f'"{arg}"')
            else:
                reconstructed.append(arg)
                
    return ' '.join(reconstructed)

from torchvision.transforms.functional import to_pil_image, to_tensor
class VaeWrapper:
    '''
    wrapper for vae that abstract the vae encode and decode proc
    '''
    def __init__(self, vae = None, type = None, pretrained_model_name_or_path= None):
        self.init_constants()
        if (vae is None) and (type is None) and ( pretrained_model_name_or_path is not None):
            #in case we got only pretrained_model_name_or_path
            self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").to("cuda")
            self.vae.requires_grad_(False)
            self.pretrained_model_name_or_path = pretrained_model_name_or_path
            self.type = [k for k,v in self.paths.items() if v==pretrained_model_name_or_path][0]

        elif (vae is not None) and (type is not None) and ( pretrained_model_name_or_path is None):
            #we got the vae and its type
            self.vae = vae
            self.type = type

        elif (vae is None) and (type is not None) and ( pretrained_model_name_or_path is None):
            #than we got only the type. from it infer the model and so on
            self.type = type
            assert self.type in ["sd1.5","rv", "sd2.1", "sd3" ,"flux","dream_like"]
            self.pretrained_model_name_or_path = self.paths[self.type]
            self.vae = AutoencoderKL.from_pretrained(self.pretrained_model_name_or_path, subfolder="vae").to("cuda")
            self.vae.requires_grad_(False)
        else:
            #recieved some combination, so to make sure, i will raise exception
            raise RuntimeError("YOU MUST CHOOSE TO EITHER PASS VAE AND TYPE OR ONLY A PATH ")

        assert self.type in ["sd1.5","rv", "sd2.1", "sd3" ,"flux","dream_like"]
        
    def init_constants(self):
        self.types = {
            "sd1.5" : (self._sd1_encode, self._sd1_decode),
            "rv" : (self._sd1_encode, self._sd1_decode),
            "dream_like": (self._sd1_encode, self._sd1_decode),
            "sd2.1" : (self._sd2_encode, self._sd2_decode),
            "sd3" : (self._sd3_encode, self._sd3_decode),
            "flux" : (self._flux_encode, self._flux_decode),
        }
        self.paths = {
            "sd1.5" : "botp/stable-diffusion-v1-5",
            "rv" : "stablediffusionapi/realistic-vision-v51",
            "dream_like" : "dreamlike-art/dreamlike-photoreal-2.0",
            "sd2.1" :"stabilityai/stable-diffusion-2-1",
            "sd3" : "stabilityai/stable-diffusion-3.5-medium",
            "flux" : None,
            }
        
        self.num_latent_channels = {
            "sd1.5" : 4,
            "rv" : 4,
            "dream_like": 4,
            "sd2.1" :4,
            "sd3" : 16,
            "flux" : 16,
            }

    def get_pretrained_model_name_or_path(self):
        return self.pretrained_model_name_or_path
    def get_vae_type(self):
        return self.type
        
    def get_num_latent_channels(self):
        return  self.num_latent_channels[self.type]
    
    def _sd1_encode(self,image):
        # assume image in range [-1,1], shape B,3,512,512
        z = self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
        return z

    def _sd1_decode(self,latent):
        # assume latent shape B,4,64,64
        image = self.vae.decode(latent / self.vae.config.scaling_factor).sample.clamp(-1,1)
        return image
    
    def _sd2_encode(self,image):
        # assume image in range [-1,1], shape B,3,512,512
        z = self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
        return z

    def _sd2_decode(self,latent):
        # assume latent shape B,4,64,64
        image = self.vae.decode(latent / self.vae.config.scaling_factor).sample.clamp(-1,1)
        return image
    
    def _sd3_encode(self,image):
        # assume image in range [-1,1], shape B,3,512,512
        latent = (self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor) - self.vae.config.shift_factor
        return latent

    def _sd3_decode(self,latent):
        # assume latent shape B,16,64,64
        latent = (latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latent, return_dict=False)[0].clamp(-1,1)
        return image
    
    def _flux_encode(self,image):
        # assume image in range [-1,1], shape B,3,512,512
        #IMPORTANT! there is an _pack stage that WE DO NOT PERFORM HERE
        latent = (self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor) - self.vae.config.shift_factor
        return latent

    def _flux_decode(self,latent):
        # assume latent shape B,16,64,64
        #IMPORTANT! there is an _unpack stage that WE DO NOT PERFORM HERE
        latent = (latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latent, return_dict=False)[0].clamp(-1,1)
        return image

    def encode(self,image,input_as="[-1,1] image"):
        with torch.no_grad():
            assert input_as in ["[-1,1] image", "[0,1] image", "pil image"]
            if input_as in ["[0,1] image", "pil image"]:
                if input_as == "pil image":
                    image = to_tensor(image).unsqueeze(0).to(self.vae.device)
                image = ((image*2) - 1).clamp(-1,1)
            return self.types[self.type][0](image)

    def decode(self,latent,return_as = "[-1,1] image"):
        with torch.no_grad():
            assert return_as in ["[-1,1] image", "[0,1] image", "pil image"]
            image = self.types[self.type][1](latent)
            if return_as in ["[0,1] image", "pil image"]:
                image = ((image/2) + 0.5 ).clamp(0,1)
                if return_as == "pil image":
                    image = to_pil_image(image[0].float())
            return image

    
# CNN
class DegradationModel(nn.Module):
    def __init__(self, latent_dim,output_dim=0,sigma_condition=False,layer_norm = False):
        super(DegradationModel, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.layer_norm = layer_norm
        self.sigma_condition = sigma_condition
        if output_dim==0:
            output_dim = latent_dim
        if sigma_condition:
            # self.sigma_condition_precondition = nn.Sequential(nn.Linear(1280, 64 * 64),nn.ReLU(),nn.Linear(64 * 64, 64 * 64))
            self.sigma_condition_precondition = nn.Sequential(nn.Linear(1280, 32),nn.LeakyReLU(),nn.Linear(32, 64 * 64))
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(latent_dim +sigma_condition, 16 +sigma_condition, kernel_size=3, stride=1, padding=1), #out b 16, 63, 63
            nn.BatchNorm2d(16 +sigma_condition) if not layer_norm else nn.LayerNorm([16 +sigma_condition, 64, 64]),
            nn.ReLU(),
            nn.Conv2d(16+sigma_condition, 32+sigma_condition, kernel_size=3, stride=1, padding=1), # out b, 32 ,62, 62
            nn.BatchNorm2d(32+sigma_condition) if not layer_norm else nn.LayerNorm([32+sigma_condition, 64, 64]),
            nn.ReLU(),
            nn.Conv2d(32+sigma_condition, 64+sigma_condition, kernel_size=3, stride=1, padding=1),# out b, 64 ,61, 61
            nn.BatchNorm2d(64+sigma_condition) if not layer_norm else nn.LayerNorm([64+sigma_condition, 64, 64]),
            nn.ReLU(),
            nn.Conv2d(64+sigma_condition, 128, kernel_size=3, stride=1, padding=1),# out b, 128 ,60, 60
            nn.BatchNorm2d(128) if not layer_norm else nn.LayerNorm([128, 64, 64]),
            nn.ReLU(),
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), #out b, 64, 
            nn.BatchNorm2d(64) if not layer_norm else nn.LayerNorm([64, 64, 64]),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32) if not layer_norm else nn.LayerNorm([32, 64, 64]),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16) if not layer_norm else nn.LayerNorm([16, 64, 64]),
            nn.ReLU(),
        )
        
        # Adjusting dimension of encoded representation
        self.adjust_encoding = nn.Sequential(
            nn.Conv2d(128+sigma_condition, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16) if not layer_norm else nn.LayerNorm([16, 64, 64]),
            nn.ReLU(),
        )
        
        # Additional processing layers after skip connection
        self.processing = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16) if not layer_norm else nn.LayerNorm([16, 64, 64]),
            nn.ReLU(),
            nn.Conv2d(16, output_dim, kernel_size=3, stride=1, padding=1),
            # nn.Tanh()  # Output in [-1, 1] range
        )

    def forward(self, z_clean,sigma_emb,**kwargs):
        if self.sigma_condition:
            sigma_emb = self.sigma_condition_precondition(sigma_emb)
            sigma_emb = einops.rearrange(sigma_emb, 'B (h w) -> B 1 h w',h=64,w=64)
            z_clean = torch.concat([z_clean,sigma_emb],dim=1)
        # Encode the clean latent representation
        encoded = self.encoder(z_clean)
        
        # Decode to obtain degraded latent representation
        decoded = self.decoder(encoded)

        if self.sigma_condition:
            encoded = torch.concat([encoded,sigma_emb],dim=1)
        # Adjust the size of the encoded representation
        adjusted_encoded = self.adjust_encoding(encoded)
        
        # Add skip connections between corresponding encoder and decoder layers
        skip_connected = decoded + adjusted_encoded
        
        # Process the skip connected output
        processed = self.processing(skip_connected)
        
        return processed

def decode(lat,vae,i=0,normalized = True,all=False):
    with torch.no_grad():
        if all:
            reconstructed_tensor = vae.decode(lat/ vae.config.scaling_factor).sample
        else:
            reconstructed_tensor = vae.decode(lat[i:i+1]/ vae.config.scaling_factor).sample
        if normalized:
            reconstructed_tensor = (reconstructed_tensor/2 + 0.5).clamp(0,1).cpu()
        return reconstructed_tensor

#from RG
def embed_timestep(unet, sample, timestep):
    timesteps = preprocess_timestep(sample, timestep)
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = unet.time_proj(timesteps)
    t_emb = t_emb.to(dtype=sample.dtype)
    emb = unet.time_embedding(t_emb, None)
    return emb

def preprocess_timestep(sample, timestep):
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)
    return timesteps

# pipeline, from diffusers code base

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class StableDiffusionPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    LoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
        sigma_condition = False,
        sampling_args = None,
        cond_inpaint = False

        
    ):
        super().__init__()

        self.dps_sampler = PosteriorSampling(operator=None, noiser=None)
        if sampling_args.model=="rg":
            from readout_guidance import rg_pipeline , rg_helpers
            from dhf import aggregation_network
            self.aggregation_network = aggregation_network.AggregationNetwork(projection_dim=384,
            feature_dims=[1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320],
            device="cuda",
            save_timestep=[0],
            num_timesteps=1000,
            use_output_head=True,
            output_head_channels=4,
            output_head_act=False,
            bottleneck_sequential=False,
            sigma_condition = sigma_condition,
            cond_inpaint = cond_inpaint
            )
        
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        if sampling_args.model=="rg":
            self.controller = rg_pipeline.ReadoutGuidance(
            self,
            edits=True,
            points=None,
            latent_dim=64,
            idxs=None
            )

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            image_embeds = []
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )
                single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
                single_negative_image_embeds = torch.stack(
                    [single_negative_image_embeds] * num_images_per_prompt, dim=0
                )

                if do_classifier_free_guidance:
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                    single_image_embeds = single_image_embeds.to(device)

                image_embeds.append(single_image_embeds)
        else:
            repeat_dims = [1]
            image_embeds = []
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                    single_negative_image_embeds = single_negative_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
                    )
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                else:
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                image_embeds.append(single_image_embeds)

        return image_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, None

    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.FloatTensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt
    

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        do_dps = False,
        w = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        sampling_args = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        set_seeds(sampling_args.seed)
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        # set_seeds(sampling_args.seed,deterministic=True)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                if hasattr(self,"sdedit_t") and t.cpu().detach().item()>self.sdedit_t:
                    continue
                        
                latents = latents.requires_grad_()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]


                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                if self.do_classifier_free_guidance and sampling_args.cfg_fix:
                    new_latents_cfg_1 = self.scheduler.step(noise_pred_text, t, latents, **extra_step_kwargs, return_dict=True)
                    pred_original_sample_cfg_1 = new_latents_cfg_1["pred_original_sample"]
                    prev_sample_cfg_1 = new_latents_cfg_1["prev_sample"]

                new_latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                pred_original_sample = new_latents["pred_original_sample"]
                prev_sample = new_latents["prev_sample"]
                
                    
                if do_dps:

                    #create sigma embedding
                    sigma = self.dps_sampler.noiser.sigma
                    sigma = (sigma/0.1)*999
                    sigma_emb = embed_timestep(self.unet, pred_original_sample, sigma)
                    
                    #each latent operator potentially has a fifferent way to create w hat
                    if sampling_args.model == "rg":
                        #collect the features from the Unet
                        feats = self.controller.collect_and_resize_feats()
                        if self.do_classifier_free_guidance: 
                            feats_uncond, feats_text = feats.chunk(2)
                            feats = feats_text

                        #create w_hat
                        feats = einops.rearrange(feats, 'b w h c -> b c w h')
                        w_hat = self.aggregation_network(feats, timestep_cond,sigma_emb=sigma_emb,cond_inpaint=kwargs["cond_inpaint"])
                    elif sampling_args.model == "cnn":
                        if self.do_classifier_free_guidance and sampling_args.cfg_fix:
                            w_hat = self.dps_sampler.operator.forward(pred_original_sample_cfg_1, sigma_emb=sigma_emb)
                        else:
                            w_hat = self.dps_sampler.operator.forward(pred_original_sample, sigma_emb=sigma_emb)
                    elif sampling_args.model == "swin_v2":
                        w_hat = self.dps_sampler.operator(pred_original_sample).reconstruction
                    elif sampling_args.model == "swin_v2_cross":
                        w_hat = self.dps_sampler.operator(pred_original_sample, measurement=w).reconstruction
                    elif sampling_args.model == "adaptive_swin":
                        w_hat = self.dps_sampler.operator(pred_original_sample, degraded_pixel_values=w)
                    elif sampling_args.model == "LatentDegradationNetwork":
                        w_hat = self.dps_sampler.operator(pred_original_sample, torch.tensor([self.dps_sampler.noiser.sigma] * pred_original_sample.shape[0]).to(pred_original_sample.device))
                    elif sampling_args.model == "preserve_LatentDegradationNetwork":
                        w_hat = self.dps_sampler.operator(pred_original_sample, torch.tensor([self.dps_sampler.noiser.sigma] * pred_original_sample.shape[0]).to(pred_original_sample.device))
                    elif "preserve_LatentDegradationNetwork" in sampling_args.model:
                        w_hat = self.dps_sampler.operator(pred_original_sample, torch.tensor([self.dps_sampler.noiser.sigma] * pred_original_sample.shape[0]).to(pred_original_sample.device))
                    
                    latents , norm = self.dps_sampler.conditioning(x_prev=latents, x_t=prev_sample, x_0_hat=pred_original_sample, w=w,w_hat=w_hat)
                    latents = latents.detach()
                else:
                    latents = prev_sample
                    latents = latents.detach()
                    norm = torch.tensor(-1.0)

                
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                progress_bar.set_postfix({"t": t.cpu().item() ,"norm": norm.cpu().item()},refresh=False)
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
                    
        latents = latents.detach()
        with torch.no_grad():
            if not output_type == "latent":
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                    0
                ]
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
                has_nsfw_concept = None
            else:
                image = latents
                has_nsfw_concept = None
    
            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
    
            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
    
            # Offload all models
            self.maybe_free_model_hooks()
    
            if not return_dict:
                return (image, has_nsfw_concept)
    
            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)


