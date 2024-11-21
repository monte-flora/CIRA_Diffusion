""" 

This is a script to train a score-based diffusion model based on Karras et al. (2022) 'EDM'.

This code is written in pytorch and is a smattering of functions from different places... so please don't hate me. 

WARNING: CURRENTLY THE CODE IS HARD CODED TO DO 1 CONDITIONAL IMAGE PREDICTION. 

You will need: 
1) Pytorch (this is the main lift here, duh)
2) Diffusers (this helps with model building... and eventually a pipeline)
3) Accelerate (this will help across GPUs if you have more than one)

Author: Randy Chase 
Date: June 2024 
Email: randy 'dot' chase 'at' colostate.edu 

Example call: 

$ accelerate launch train_diffusion_model_EDM_clean.py 

TODO:: 
1) Find the right pipeline in diffusers that is the same as the EDM sampler below. 
2) Generalize classes to do more than one image prediction. 

"""

#################### Imports ########################
# Im sure there is an extra import or two, but havent cleaned it up yet. 
import os

import wandb

os.environ["WANDB__SERVICE_WAIT"] = "300"

# Set the OPENBLAS_NUM_THREADS environment variable
#os.environ["OPENBLAS_NUM_THREADS"] = "1"


from dataclasses import dataclass
import dataclasses
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDPMScheduler#, DiTTransformer2DModel
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
import gc
 
import torch.distributed as dist
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path

import math
import time

from diffusers.pipelines import DiffusionPipeline,ImagePipelineOutput
from typing import List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor
import wandb

#################### \Imports ########################


#################### Classes ########################

# num_steps = (n_samples / batch_size) * epochs

@dataclass
class TrainingConfig:
    """ This should be probably in some sort of config file, but for now its here... """

    model_arch = 'unet2d' # DiT = DiffusionTransformer
    
    image_size = 160  # the generated image resolution, which is the same size of my training data, note it has to be square. 
    train_batch_size = 100 #this was as manny batch,7,256,256 images i could fit in the 95 GB of RAM 

    # For the full model, batch size of 50, but 100 for just refl.
    n_channels = 1 
    n_gpus = 2
    num_epochs = 4000 #how long to train, this is about where 'convergence' happened and takes 2 hours per epoch on 1 GPU. 
    gradient_accumulation_steps = 1 # I havent gotting this working yet....
    learning_rate = 1e-4 #I am using a learning rate scheduler, not sure this is even used?
    lr_warmup_steps = 3500 #not sure if a warmup is needed, but just left it 
    save_model_epochs = 1 #save the model every epoch, just in case things DIE 
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision 

    #data_file = '/work/mflora/wofs-cast-data/predictions/wofscast_normalized_with_residual_160_samples.pt'
    data_file = '/ourdisk/hpc/ai2es/wofscast/wofscast_normalized_with_residual_10K_samples.pt'
    #data_file = '/ourdisk/hpc/ai2es/wofscast/wofscast_normalized_with_residual_160_samples.pt'

    
    output_dir = '/ourdisk/hpc/ai2es/wofscast/diffusion_model_ckpts/diffusion_refl_only'
    #output_dir = '/work2/mflora/wofscast_diffusion_model_ckpts/diffusion_refl_only_transformer'
    
    push_to_hub = False # whether to upload the saved model to the HF Hub, i havent tested this 
    hub_private_repo = False # or this 
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook 
    seed = 0 #random seed 
    
   
class WoFSCastDataset(Dataset):
    
    """
    Pytorch Dataset class for loading the WoFSCast predictions 
    and the associated WoFS target fields. 
    """ 
    # next = WoFS
    # cond = WoFSCast
    
    def __init__(self, 
                 wofscast_predictions, 
                 target_residuals, 
                 diffs_stddev_path=None,
                 variable_indices=None,
                 metadata=None):
        
       
        self.wofscast_predictions = wofscast_predictions
        self.target_residuals = target_residuals
        self.variable_indices = variable_indices
        self.metadata = metadata

    def __len__(self):
        return len(self.target_residuals)

    def __getitem__(self, index):
        target_residuals = self.target_residuals[index]
        wofscast_predictions = self.wofscast_predictions[index]

        # Select a subset of variables. E.g., selecting
        # only composite reflectivity. 
        if self.variable_indices:
            target_residuals = target_residuals[self.variable_indices]
            wofscast_predictions = wofscast_predictions[self.variable_indices]

        # DEPRECATED. NEW DATASETS ALREADY HAVE THE RESIDUALS!!
        # Predicting the residual similar to CorrDiff
        # residual = wofs - wofscast
        # At sample time, wofscast + residual ~ wofs 
        #residual = target_residuals - wofscast_predictions

        if self.metadata is not None:
            metadata = self.metadata[index]
            return wofscast_predictions, target_residuals, metadata
        else:
            return wofscast_predictions, target_residuals
    
class EDMPrecond(torch.nn.Module):
    """ Original Func:: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/networks.py#L519
    
    This is a wrapper for your pytorch model. It's purpose is to apply the preconditioning that is talked about in Karras et al. (2022)'s EDM paper. 
    
    I adapted the linked function to take a conditional input. Note for now, the condition is concatenated to the dimension you want denoise (dim:0 for one channel prediction). 
    
    
    
    """
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # idk if this is still needed..?
        model,                              # pytorch model from diffusers 
        label_dim       = 0,                # Ignore this for now 
        use_fp16        = True,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data. this was the default from above
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, inputs, noisy_targets, sigma, force_fp32=False, **model_kwargs):
        
        """ 
        Call method. Preconditioning model from the Karras EDM Paper. 
        
        inputs : PyTorch Tensor of shape (batch, n_channels, ny, nx)
            Conditional input images. 
        targets : PyTorch Tensor of shape (batch, n_channels, ny, nx) 
            Target images with noise-level sigma applied. 
        """
        # To follow the naming convention from RJC and Karras original code. 
        x_condition = inputs.to(torch.float32)
        x_noisy = noisy_targets.to(torch.float32) 
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        
        #forcing dtype matching?
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and noisy_targets.device.type == 'cuda') else torch.float32
        
        #get weights from EDM 
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        #concatenate back with the scaling applied to the noisy image 
        model_input_images = torch.cat([x_noisy*c_in, x_condition], dim=1)
        
        #denoise the image (e.g., run it through your diffusers model) 
        F_x = self.model((model_input_images).to(dtype), c_noise.flatten(),
                         #class_labels=torch.zeros(model_input_images.shape[0], 
                         #                         dtype=torch.long).to(model_input_images.device),
                         return_dict=False)[0]
 
        #is this needed? RJC 
        assert F_x.dtype == dtype
        #do scaling from EDM 
        D_x = c_skip * x_noisy + c_out * F_x.to(torch.float32)
        
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class EDMLoss:
    
    """Original Func:: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py
    
    This is the loss function class from Karras et al. (2022)'s EDM paper. Only thing changed here is that the __call__ takes the clean_images and the condition_images seperately. It expects your model to be wrapped with that EDMPrecond class. 
    
    """
    def __init__(self, P_mean=-1.2,  
                 P_std=1.2, 
                 sigma_data=0.5, # MLF: Changed from 0.5 -> 1 to match GenCast 
                 ):
        """ These are the defaults from the paper, 'worked' for my simple 10 min example forecast """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
         
    def __call__(self, net, inputs, targets, labels=None, augment_pipe=None):
        """ 
        
        Apply a denoiser neural network to a combined noisy target and 
        clean conditional input to get a denoised target field. 
        Compute loss between denoised target field and original target field.
                
        net : PyTorch Module  
            Model wrapped with EDMPrecond; the "Denoiser"
        inputs : PyTorch Tensor of shape (batch, n_channels, ny, nx) 
            Conditional input images
        targets : PyTorch Tensor of shape (batch, n_channels, ny, nx)
            Target images
        """
        #get random seeds, one for each image in the batch 
        rnd_normal = torch.randn([targets.shape[0], 1, 1, 1], device=targets.device)
        #get random noise levels
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        #get the loss weight for those sigmas 
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        #make the noise scalars images 
        n = torch.randn_like(targets) * sigma
        #add noise to the clean images 
        noisy_targets = torch.clone(targets + n)
        #cat the images for the wrapped model call 
        #model_input_images = torch.cat([noisy_images, condition_images], dim=1)
        #call the EDMPrecond model [noisy_targets, inputs]
        denoised_targets = net(inputs, noisy_targets, sigma)
        #calc the weighted loss at each pixel, the mean across all GPUs and pixels is in the main train_loop 
        loss = weight * ((denoised_targets - targets) ** 2)
        
        return loss 
        
#################### \Classes ########################

#################### Funcs ########################

def modify_path_if_exists(original_path):
    """
    Modifies the given file path by appending a version number if the path already exists.
    Useful for not overwriting existing version of the WoFSCast model parameters. 

    Args:
        original_path (str): The original file path.

    Returns:
        str: A modified file path if the original exists, otherwise returns the original path.
    """
    # Check if the file exists
    if not os.path.exists(original_path):
        return original_path

    # Split the path into directory, basename, and extension
    directory, filename = os.path.split(original_path)
    basename, extension = os.path.splitext(filename)

    # Iteratively modify the filename by appending a version number until an unused name is found
    version = 1
    while True:
        new_filename = f"{basename}_v{version}{extension}"
        new_path = os.path.join(directory, new_filename)
        if not os.path.exists(new_path):
            return new_path
        version += 1



def train_loop(config, model, optimizer, train_dataloader, lr_scheduler):
    """ 
    This is the main show! the training loop 
    """
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
    elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
   
   
    if accelerator.is_main_process: 
        accelerator.init_trackers(
            project_name = "wofscast-gen", 
            init_kwargs = {"wandb" : {"name" : os.path.basename(config.output_dir), 
                                     "config" : dataclasses.asdict(config),
                                  } 
            }
            )

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    #iterator to see how many gradient steps have been done
    global_step = 0
    
    # Define parameters for early stopping, TODO: this needs to be in the config step 
    patience = 40  # Number of epochs to wait for improvement
    min_delta = 1e-6  # Minimum change in loss to be considered as improvement
    best_loss = float('inf') #fill with inf to start 
    no_improvement_count = 0
    window_size = 5  # Define the window size for the moving average
    loss_history = [] # Initialize a list to store the recent losses
    
    #define loss, you can change the sigma vals here (i.e., hyperparameters) TODO: add to config... 
    loss_fn = EDMLoss()
    
    # Now you train the model
    for epoch in range(config.num_epochs):
        #this is for the cmd line
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        #initalize loss to keep track of the mean loss across all batches in this epoch 
        epoch_loss = torch.tensor(0.0, device=accelerator.device)
        
        for step, (inputs, targets) in enumerate(train_dataloader):
            
            #this is the autograd steps within the .accumulate bit (this is important for multi-GPU training)
            with accelerator.accumulate(model):
                
                #send data into loss func and get the loss (the model call is in here)
                per_sample_loss = loss_fn(model, inputs, targets)
                
                #in the loss_fn, it returns the squarred error (per pixel basis), we need the mean for the loss  
                loss = per_sample_loss.mean()
                
                #calc backprop 
                accelerator.backward(loss)
                
                #clip gradients (is this needed? idk leftover from Butterflies example)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                #step 
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            
            # Accumulate epoch loss on each GPU seperately 
            epoch_loss += loss.detach()
            
            #log things on tensorboard 
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # Synchronize epoch loss across devices, this will just concat the two 
        epoch_loss = accelerator.gather(epoch_loss)

        # Sum up the losses across all GPUs
        total_epoch_loss = epoch_loss.sum()

        # the batches are split from the train_dataloader to each GPU
        total_samples_processed = len(train_dataloader) * accelerator.num_processes

        # Calculate mean epoch loss by dividing by the total number of batches proccessed 
        mean_epoch_loss = total_epoch_loss / total_samples_processed
        
        # Print or log the average epoch loss, need to convert to scalar to get tensorboard to work (using .item())
        logs = {"epoch_loss": mean_epoch_loss.item(), "epoch": epoch}
        accelerator.log(logs, step=epoch)
        
        #accumulate rolling mean 
        loss_history.append(mean_epoch_loss.item())
        
        # Calculate the moving average if enough epochs have passed
        if len(loss_history) >= window_size:
            moving_average = sum(loss_history[-window_size:]) / window_size
            logs = {"moving_epoch_loss": moving_average, "epoch": epoch}
            accelerator.log(logs, step=epoch)

            # Check for improvement in the moving_average
       #     if moving_average < (best_loss - min_delta):
       #         best_loss = moving_average
       #         no_improvement_count = 0
       #     else:
       #         no_improvement_count += 1
        
        # This is the eval and saving step 
        if accelerator.is_main_process:
            #This will be used when i get the right Diffusers pipeline. For now we just store the model 
            #pipeline = DDPMCondPipeline2(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler).to("cuda:0")
            
            #this is if you want to output images... not supported currently 
            # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                # evaluate(config, epoch, pipeline)
                
            #this is to save the model 
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    #need to grab the unwrapped diffusers model from EDMPrecond 
                    accelerator.unwrap_model(model).model.save_pretrained(config.output_dir)

        # MLF: Comment out the early stopping procedure. 
        # Check if training should be stopped due to lack of improvement
        #if no_improvement_count >= patience:
        #    print(f"Early stopping triggered after {patience} epochs without improvement.")
        #    print(f"Killing processes using dist.barrier() and dist.destroy_process_group()")
        #    # Signal all processes to stop
        #    dist.barrier()  # Ensure all processes are synchronized
        #    dist.destroy_process_group()  # Properly shut down distributed training
        #    break
                    
        gc.collect()
        
        
    accelerator.end_training()   


#################### \Funcs ########################

#################### CODE ########################

#initalize config·
config = TrainingConfig()

n_channels = config.n_channels # Hardcoded by MLF
n_gpus = config.n_gpus # Hardcoded by MLF;used to compute number of training steps

config.output_dir = modify_path_if_exists(config.output_dir)

print(f'Saving data to {config.output_dir=}')

data_file = config.data_file 

# Load the saved dataset from disk, this will take a min depending on the size 
print('\n Loading dataset...\n')
start_time = time.time()
dataset = torch.load(data_file, )
print(f'Data Loading time: {time.time()-start_time:.3f} secs')

# Only loading the composite reflectivity images
if n_channels == 1:
    variable_indices = [0]
else:
    variable_indices= None

torch_dataset = WoFSCastDataset(wofscast_predictions=dataset['input_images'], 
                                target_residuals=dataset['target_images'],
                                variable_indices = variable_indices,
                               )
n_samples = len(torch_dataset)

print(f'{len(torch_dataset)=}')

#throw it in a dataloader for fast CPU handoffs. 
#Note, you could add preprocessing steps with image permuations here i think 
train_dataloader = torch.utils.data.DataLoader(torch_dataset, 
                                               batch_size=config.train_batch_size,
                                               shuffle=True)

if config.model_arch == 'unet2d':

    #go ahead and build a UNET, this was the exact same as the butterfly example, but different channels. This is a big model.. 
    # in_channels = noisy_dim + condition_channels. 
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=2*n_channels,  # input and target set of images.
        out_channels=1*n_channels,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        #block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        block_out_channels = (128, 128, 256, 256, 256, 256), 
        downsample_type='resnet', 
        upsample_type='resnet', # Resnet rather than conv 
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        dropout=0.1, 
        )
elif config.model_arch == 'dit':
    model = DiTTransformer2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=2*n_channels,  # input and target set of images.
        out_channels=1*n_channels,  # the number of output channels
        num_layers = 6,
        patch_size=8 
    ) 
    
# MLF: Matching the sigma args to match GenCast
model_wrapped = EDMPrecond(config.image_size, n_channels, model,)  
                           #sigma_min=0.002, 
                           #sigma_max=1000, 
                           #sigma_data=0.5)

total_params = sum(p.numel() for p in model_wrapped.parameters())
print(f"\n\nNumber of parameters: {total_params/1_000_000:.2f} Million\n\n")

#left this the same as the butterfly example 
optimizer = torch.optim.AdamW(model_wrapped.model.parameters(), 
                              lr=config.learning_rate, 
                              weight_decay=0.0001 # MLF: Add to match GenCast
                             )

num_steps = (n_samples / (n_gpus*config.train_batch_size)) * config.num_epochs

print(f'{num_steps=}')

lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=num_steps,
    )

#main method here! 
train_loop(config, model_wrapped, optimizer, train_dataloader, lr_scheduler)
