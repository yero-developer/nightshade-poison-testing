import copy

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from utils.data_manipulation import linear_beta_schedule
from utils.dataloaders import cifar10_32x32_loader, cifar100_32x32_loader
from utils.train import training

from diffusion_networks.unet_diffusion_model_1 import Unet_Diffusion_Model_1


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

torch.use_deterministic_algorithms(False, warn_only=True)

torch.set_float32_matmul_precision('medium')

'''
torch._dynamo.config.suppress_errors = True  resolves issues with torch.compile() failing
Many warnings will be shown but torch.compile will run properly
'''
torch._dynamo.config.suppress_errors = True
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.cudagraph_trees = False



device = "cuda" if torch.cuda.is_available() else "cpu"

class EMA:
    def __init__(self, model, decay=0.9999, device=None):
        # Deepcopy model weights for EMA initialization
        self.ema_model = copy.deepcopy(model)
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema_model.to(device)
        self.ema_model.eval()
        # Disable grad for EMA model
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, decay):
        self.decay = decay
        msd = model.state_dict()
        esd = self.ema_model.state_dict()
        for k in esd.keys():
            if k in msd:
                # Smoothly update EMA weights
                esd[k].mul_(self.decay).add_(msd[k], alpha=1 - self.decay)
            else:
                # Optionally warn once, or just silently skip
                print(f"Skipping key {k} in EMA update; not found in model.")
                pass

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)

    def to(self, device):
        self.ema_model.to(device)

    def copy_to(self, model):
        model.load_state_dict(self.ema_model.state_dict())

def cifar10_32x32_run(epochs_limit, poison, continue_train, model_name):
    train_dataset, test_dataset = cifar10_32x32_loader(poison)
    model = Unet_Diffusion_Model_1(input_resolution=(32, 32), time_emb_dim=64, num_classes=10).to(device)

    try:
        model_c = torch.compile(model, mode='default')
    except Exception as e:
        model_c = model

    ema = EMA(model_c, decay=0.9999, device=device)
    print("This model has", sum(p.numel() for p in model.parameters()), "parameters.")

    accumulation_steps = 1
    train_batch_size = 128
    test_batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                             num_workers=16, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True,
                              num_workers=4, persistent_workers=True)
    max_lr = 1e-3

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,
        T_mult=2,
        eta_min=1e-6
    )

    criterion = nn.MSELoss()
    timesteps_int = 1000

    betas = linear_beta_schedule(timesteps_int).to(device)

    training(model_name, train_loader, test_loader, model, ema,
             optimizer, scheduler, criterion, accumulation_steps, epochs_limit, train_batch_size, device, timesteps_int,
             betas, continue_train)


def cifar100_32x32_run(epochs_limit, poison, continue_train, model_name):
    train_dataset, test_dataset = cifar100_32x32_loader(poison)
    model = Unet_Diffusion_Model_1(input_resolution=(32, 32), time_emb_dim=128, num_classes=100).to(device)

    try:
        model_c = torch.compile(model, mode='default')
    except Exception as e:
        model_c = model


    ema = EMA(model_c, decay=0.9999, device=device)
    print("This model has", sum(p.numel() for p in model.parameters()), "parameters.")

    accumulation_steps = 1
    train_batch_size = 128
    test_batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                             num_workers=16, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True,
                              num_workers=4, persistent_workers=True)
    max_lr = 1e-3

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,
        T_mult=2,
        eta_min=1e-6
    )

    criterion = nn.MSELoss()
    timesteps_int = 1000

    betas = linear_beta_schedule(timesteps_int).to(device)

    training(model_name, train_loader, test_loader, model, ema,
             optimizer, scheduler, criterion, accumulation_steps, epochs_limit, train_batch_size, device, timesteps_int,
             betas, continue_train)


# Call one of the above functions to train a model.
cifar10_32x32_run(epochs_limit=1500,
                  poison=False,
                  continue_train=False,
                  model_name='CIFAR10_normal')



#
# cifar100_32x32_run(epochs_limit=375,
#                   poison=False, # need to add the poison stuff to dataloader properly, move the posion chairs and chosen
#                   continue_train=False,
#                   model_name='CIFAR100_normal')



