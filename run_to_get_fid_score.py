import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
import os
from diffusion_networks.unet_diffusion_model_1 import Unet_Diffusion_Model_1
from utils.data_manipulation import p_sample, linear_beta_schedule
from tqdm import tqdm
import copy

from utils.dataloaders import cifar100_32x32_loader, cifar10_32x32_loader

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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cifar10_fid_score_generate(poison, model_name):
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Transformation for real data (match InceptionV3 input)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Needed for Inception
        transforms.ToTensor(),
    ])
    batch_size = 1000


    train_dataset, test_dataset = cifar10_32x32_loader(poison)
    real_dataset = ConcatDataset([train_dataset, test_dataset])
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)


    print('Collecting real images from CIFAR10 dataset')
    # Collect real images
    for img, _ in tqdm(real_loader):
        img = img.to(device)
        img = (img * 255).clamp(0, 255).to(torch.uint8)
        fid.update(img, real=True)

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


    timesteps_int = 1000
    betas = linear_beta_schedule(timesteps_int).to(device)
    T = len(betas)

    shape = (batch_size, 3, 32, 32)

    guidance_scale = 3.5
    model.eval()

    print('Creating the fake images')
    for _ in tqdm(range(len(real_loader))):
        x = torch.randn(shape).to(device, non_blocking=True)
        class_label = torch.randint(0, 10, (batch_size,)).to(device, non_blocking=True)

        with torch.inference_mode():
            for t in reversed(range(T)):
                t_tensor = torch.full((shape[0],), t, device=device)

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        # Run unconditional pass
                        null_labels = None  # or a special null token, or an empty tensor
                        eps_uncond = model(x, t_tensor, null_labels)

                        # Run conditional pass
                        eps_cond = model(x, t_tensor, class_label)

                        predicted_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

                x = p_sample(x, t, predicted_noise, betas)


        fake_imgs = (x + 1) / 2
        fake_imgs = torch.nn.functional.interpolate(fake_imgs, size=(299, 299), mode='bilinear')
        fake_imgs = (fake_imgs * 255).to(torch.uint8)
        fid.update(fake_imgs, real=False)

    fid_score = fid.compute()
    if poison:
        print("CIFAR10 Poison model FID score:", fid_score)
    else:
        print("CIFAR10 Normal model FID score:", fid_score)
    return fid_score


def cifar100_fid_score_generate(poison, model_name):
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Transformation for real data (match InceptionV3 input)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Needed for Inception
        transforms.ToTensor(),
    ])
    batch_size = 1000


    train_dataset, test_dataset = cifar100_32x32_loader(poison)
    real_dataset = ConcatDataset([train_dataset, test_dataset])
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)

    print('Collecting real images from CIFAR10 dataset')
    # Collect real images
    for img, _ in tqdm(real_loader):
        img = img.to(device)
        img = (img * 255).clamp(0, 255).to(torch.uint8)
        fid.update(img, real=True)

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


    timesteps_int = 1000
    betas = linear_beta_schedule(timesteps_int).to(device)
    T = len(betas)

    shape = (batch_size, 3, 32, 32)

    guidance_scale = 3.5
    model.eval()

    print('Creating the fake images')
    for _ in tqdm(range(len(real_loader))):
        x = torch.randn(shape).to(device, non_blocking=True)
        class_label = torch.randint(0, 10, (batch_size,)).to(device, non_blocking=True)

        with torch.inference_mode():
            for t in reversed(range(T)):
                t_tensor = torch.full((shape[0],), t, device=device)

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        # Run unconditional pass
                        null_labels = None  # or a special null token, or an empty tensor
                        eps_uncond = model(x, t_tensor, null_labels)

                        # Run conditional pass
                        eps_cond = model(x, t_tensor, class_label)

                        predicted_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

                x = p_sample(x, t, predicted_noise, betas)


        fake_imgs = (x + 1) / 2
        fake_imgs = torch.nn.functional.interpolate(fake_imgs, size=(299, 299), mode='bilinear')
        fake_imgs = (fake_imgs * 255).to(torch.uint8)
        fid.update(fake_imgs, real=False)

    fid_score = fid.compute()
    if poison:
        print("CIFAR10 Poison model FID score:", fid_score)
    else:
        print("CIFAR10 Normal model FID score:", fid_score)
    return fid_score



def cifar10_fid_run(model_name_normal, model_name_poison):
    fid_score_normal = cifar10_fid_score_generate(poison=False,
                                                  model_name=model_name_normal)
    fid_score_poison = cifar10_fid_score_generate(poison=True,
                                                  model_name=model_name_poison)

    folder_placement = f'fid_scores_compared/'
    os.makedirs(folder_placement, exist_ok=True)

    lines_to_write = [f"CIFAR10 Normal model FID score: {fid_score_normal}\n", f"CIFAR10 Poison model FID score: {fid_score_poison}"]
    with open('fid_scores_compared/cifar10_fid_scores_compared.txt', 'w') as file:
        file.writelines(lines_to_write)

def cifar100_fid_run(model_name_normal, model_name_poison):
    fid_score_normal = cifar100_fid_score_generate(poison=False,
                                                  model_name=model_name_normal)
    fid_score_poison = cifar100_fid_score_generate(poison=True,
                                                  model_name=model_name_poison)

    folder_placement = f'fid_scores_compared/'
    os.makedirs(folder_placement, exist_ok=True)

    lines_to_write = [f"CIFAR100 Normal model FID score: {fid_score_normal}\n", f"CIFAR100 Poison model FID score: {fid_score_poison}"]
    with open('fid_scores_compared/cifar100_fid_scores_compared.txt', 'w') as file:
        file.writelines(lines_to_write)


# Run the function with the proper models that is normal and poison versions to get the FID scores.
cifar10_fid_run(model_name_normal='',
                model_name_poison='')

# cifar100_fid_run(model_name_normal='',
#                  model_name_poison='')