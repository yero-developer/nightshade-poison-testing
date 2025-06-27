import copy
import os
import torch
import random
import numpy as np

from diffusion_networks.unet_diffusion_model_1 import Unet_Diffusion_Model_1

from utils.data_manipulation import linear_beta_schedule

from utils.dataloaders import cifar10_32x32_loader, cifar100_32x32_loader

from utils.inference import inferencing


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

torch.use_deterministic_algorithms(False, warn_only=True)  # this is repeated two more times in the warning/profiler blocks

torch.set_float32_matmul_precision('medium')

'''
torch._dynamo.config.suppress_errors = True  resolves issues with torch.compile() failing
Many warnings will be shown but torch.compile will run properly
'''
torch._dynamo.config.suppress_errors = True
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.cudagraph_trees = False


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

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


def standard_cifar10_32x32_run(model_name, is_this_poison):
    train_dataset, test_dataset = cifar10_32x32_loader(False)
    model = Unet_Diffusion_Model_1(input_resolution=(32, 32), time_emb_dim=64, num_classes=10).to(device)

    model_path = os.path.join("models_checkpointed", f"{model_name}.pt")
    if os.path.isfile(model_path):
        print(f'Loading saved model {model_name} to use...')
        checkpoint = torch.load(model_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        compiled_mode = ''
        try:
            model = torch.compile(model, mode='default')
            compiled_mode = 'default'
        except Exception as e:
            model = model
            compiled_mode = 'no_compile'
        print(f'The model using compiled mode: {compiled_mode}')

        ema = EMA(model, decay=0.9999, device=device)
        ema.load_state_dict(checkpoint['ema'])
        print('Saved model is loaded.')
    else:
        print('There was an issue loading the model, please check for correct spelling or placement of the model.')

    print(train_dataset.class_to_idx)
    for k, v in train_dataset.class_to_idx.items():
        shape = (10, 3, 32, 32)
        timesteps_int = 1000

        betas = linear_beta_schedule(timesteps_int).to(device)

        target_class_name = k
        print(f'selected {target_class_name}')

        class_label = train_dataset.class_to_idx[target_class_name]

        inferencing(ema.ema_model, is_this_poison, betas, class_label, target_class_name, shape, device)

def standard_cifar100_32x32_run(model_name, is_this_poison):
    train_dataset, test_dataset = cifar100_32x32_loader(False)
    model = Unet_Diffusion_Model_1(input_resolution=(32, 32), time_emb_dim=128, num_classes=100).to(device)

    model_path = os.path.join("models_checkpointed", f"{model_name}.pt")
    if os.path.isfile(model_path):
        print(f'Loading saved model {model_name} to use...')
        checkpoint = torch.load(model_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        compiled_mode = ''
        try:
            model = torch.compile(model, mode='default')
            compiled_mode = 'default'
        except Exception as e:
            model = model
            compiled_mode = 'no_compile'
        print(f'The model using compiled mode: {compiled_mode}')

        ema = EMA(model, decay=0.9999, device=device)
        ema.load_state_dict(checkpoint['ema'])
        print('Saved model is loaded.')
    else:
        print('There was an issue loading the model, please check for correct spelling or placement of the model.')

    print(train_dataset.class_to_idx)
    for k, v in train_dataset.class_to_idx.items():
        shape = (10, 3, 32, 32)
        timesteps_int = 1000

        betas = linear_beta_schedule(timesteps_int).to(device)

        target_class_name = k
        print(f'selected {target_class_name}')

        class_label = train_dataset.class_to_idx[target_class_name]

        inferencing(ema.ema_model, is_this_poison, betas, class_label, target_class_name, shape, device)


# Run the functions one at a time, this is for the RNG to be the same for proper comparisons.
standard_cifar10_32x32_run(model_name='',
                           is_this_poison = False) # This is just for naming images as normal or poisoned version.

# standard_cifar100_32x32_run(model_name='',
#                             is_this_poison=False) # This is just for naming images as normal or poisoned version.
#
