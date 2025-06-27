import torch
import matplotlib
matplotlib.use("Agg")
import numpy as np
import os
import torchvision.transforms as transforms
from utils.data_manipulation import (p_sample)


def re_alter_image_list(images):
    images = [frame.permute(1, 2, 0) for frame in images]
    images = [frame.cpu().detach().numpy() for frame in images]

    images = [
        (np.nan_to_num(np.clip(img, 0.0, 1.0)) * 255).astype(np.uint8)
        for img in images
    ]
    return images


def inferencing(model, is_this_poison, betas, class_label, target_class_name, shape, device):
    T = len(betas)

    images = torch.randn(shape, device=device)
    x = images.to(device, non_blocking=True)
    class_label = torch.tensor(class_label).unsqueeze(0).repeat(shape[0])
    class_label = class_label.to(device, non_blocking=True)

    guidance_scale = 3.5

    model.eval()
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


    normalized_x = (x + 1) / 2
    to_pil = transforms.ToPILImage()

    folder_placement = f'samples_generated/{target_class_name}/'
    os.makedirs(folder_placement, exist_ok=True)
    for i, img in enumerate(normalized_x):
        pil_img = to_pil(img.cpu())
        if is_this_poison:
            file_name = f'{folder_placement}/{target_class_name}_{i}_poison.png'
        else:
            file_name = f'{folder_placement}/{target_class_name}_{i}_normal.png'
        pil_img.save(file_name)
