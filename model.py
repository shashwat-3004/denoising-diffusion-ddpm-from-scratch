"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    ans = torch.linspace(beta_start,beta_end, T)
    return ans

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    # TODO: return 1 - betas
    return 1-betas

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    # TODO: cumulative product of alphas
    return torch.cumprod(alphas, dim=0)

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    batch_size = x.shape[0]
    elem = a[t]
    return elem.reshape(batch_size,1,1,1)

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    # TODO: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise
    t_upd = t.reshape(x0.shape[0],1,1,1)

    alpha_bar_t = alphas_cumprod[t_upd]

    x_t = alpha_bar_t**0.5 * x0 + (1-alpha_bar_t)**0.5 * noise
    return x_t

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    # TODO: build betas, alphas, alphas_cumprod and useful sqrts
    betas = linear_beta_schedule(T, beta_start, beta_end)
    alphas = alphas_from_betas(betas)

    alpha_cumpord = cumprod_alphas(alphas)

    sqrt_alphas_cumprod = alpha_cumpord**0.5

    sqrt_one_minus_alphas_cumprod = (1-alpha_cumpord)**0.5

    s = {"T":T, "betas":betas, "alphas":alphas, "alphas_cumprod":alpha_cumpord, "sqrt_alphas_cumprod":sqrt_alphas_cumprod, "sqrt_one_minus_alphas_cumprod":sqrt_one_minus_alphas_cumprod}
    return s

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    # TODO: MSE between predicted and true noise
    loss = noise - noise_pred

    return ((noise-noise_pred)**2).mean()

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    # TODO: q_sample -> model -> MSE(noise_pred, noise)
    
    x_t = alphas_cumprod[t]**0.5 * x0 + (1-alphas_cumprod[t])**0.5 * noise

    noise_pred = model(x_t,t)
    return noise_prediction_loss(noise_pred,noise)

# Step 9 - timestep_embedding
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    # TODO: sinusoidal timestep embedding of shape (B, dim)
    indexes = torch.arange(0,dim//2,1)

    freq = (1/10000**(2*indexes/dim)) if dim//2 != 1 else torch.tensor(1)

    embedding = torch.zeros(t.shape[0], dim)

    embedding[...,0:dim//2] = torch.sin(t.unsqueeze(1)/freq.unsqueeze(0))
    embedding[...,dim//2:] = torch.cos(t.unsqueeze(1)/freq.unsqueeze(0))
    return embedding

# Step 10 - init_tiny_unet (not yet solved)
# TODO: implement

# Step 11 - tiny_unet_forward (not yet solved)
# TODO: implement

# Step 12 - make_blob_dataset (not yet solved)
# TODO: implement

# Step 13 - ddpm_train_step (not yet solved)
# TODO: implement

# Step 14 - train_ddpm (not yet solved)
# TODO: implement

# Step 15 - predict_x0_from_eps (not yet solved)
# TODO: implement

# Step 16 - ddpm_p_mean_variance (not yet solved)
# TODO: implement

# Step 17 - ddpm_p_sample (not yet solved)
# TODO: implement

# Step 18 - ddpm_sample_loop (not yet solved)
# TODO: implement

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

