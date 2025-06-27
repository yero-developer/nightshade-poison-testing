import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import math
import random
import numpy as np


def get_subset_loader(dataset, subset_size, batch_size):
    indices = random.sample(range(len(dataset)), subset_size)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16, persistent_workers=True)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps + s) / (1 + s)) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)  # tighter clipping helps with stability
    return betas


def q_sample(x_start, t, noise, betas):
    device = x_start.device
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).to(device)


    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1)
    sqrt_1m_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1)

    return sqrt_alpha_cumprod_t * x_start + sqrt_1m_alpha_cumprod_t * noise


def p_sample(x_t, t, pred_noise, betas):
    if isinstance(t, int):
        t = torch.tensor([t], device=x_t.device, dtype=torch.long).expand(x_t.shape[0])

    device = x_t.device
    B = x_t.shape[0]
    alphas = 1. - betas

    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas = torch.sqrt(alphas).gather(0, t).view(B, 1, 1, 1).to(device)
    sqrt_recip_alpha = torch.sqrt(1. / alphas).gather(0, t).view(B, 1, 1, 1).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod.to(device)).gather(0, t).view(B, 1, 1, 1)
    beta_t = betas.gather(0, t).view(B, 1, 1, 1)

    mean = (1. / sqrt_alphas) * (x_t - (beta_t / sqrt_one_minus_alphas_cumprod) * pred_noise)

    if (t > 0).any():
        # Variance
        alpha_cumprod_t = alphas_cumprod.gather(0, t).view(B, 1, 1, 1)
        alpha_cumprod_prev = alphas_cumprod.gather(0, torch.clamp(t - 1, 0)).view(B, 1, 1, 1)
        posterior_variance = beta_t * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod_t)
        sigma = torch.sqrt(posterior_variance + 1e-8)

        noise = torch.randn_like(x_t).to(device)
        return mean + sigma * noise
    else:
        return mean



