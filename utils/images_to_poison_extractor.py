import os
import shutil
from torchvision import datasets
from tqdm import tqdm
from pathlib import Path


def save_images_by_class(dataset, count, root_dir):
    for idx, (img, label) in enumerate(tqdm(dataset, desc=f"Saving to {root_dir}")):
        class_name = dataset.classes[label]
        class_dir = os.path.join(root_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, f"{class_name}_{count}.png"))  # img is already a PIL image
        count += 1
    return count

def cifar10_images_to_poison(project_root):
    data_dir = os.path.join(project_root, "./data")
    output_dir = os.path.join(project_root, "cifar10_images")

    # Download CIFAR-10 datasets (train and test)
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True)

    # Save images by class
    count = 0

    count = save_images_by_class(train_set, count, output_dir)
    count = save_images_by_class(test_set, count, output_dir)

    print(f"CIFAR-10 images saved by class in: {output_dir}, count: {count}")

    chosen_dogs = []
    with open(str(project_root) + '/utils/chosen_dogs.txt', 'r') as f:
        dog_pics = f.readlines()
        for dog_pic in dog_pics:
            chosen_dogs.append(dog_pic.split('\n')[0])

    dog_folder_dir = Path(project_root / "cifar10_images/dog")

    dog_chosen_folder_dir = Path(project_root / 'poisoned_folder/dog_chosen')
    dog_chosen_folder_dir.mkdir(parents=True, exist_ok=True)

    for f in dog_folder_dir.iterdir():
        if f.is_file() and f.name.split('.')[0] in chosen_dogs:
            shutil.copy2(f, dog_chosen_folder_dir / f.name)
    print(f'CIFAR-10 images to be poisoned saved in: /poisoned_folder/dog_chosen folder')

def cifar100_images_to_poison(project_root):
    data_dir = os.path.join(project_root, "./data")
    output_dir = os.path.join(project_root, "cifar100_images")

    # Download CIFAR-100 datasets (train and test)
    train_set = datasets.CIFAR100(root=data_dir, train=True, download=True)
    test_set  = datasets.CIFAR100(root=data_dir, train=False, download=True)

    # Save images by class
    count = 0

    count = save_images_by_class(train_set, count, output_dir)
    count = save_images_by_class(test_set, count, output_dir)

    print(f'CIFAR-100 images saved by class in: {output_dir}, count: {count}')

    chosen_chairs = []
    with open(str(project_root) + '/utils/chosen_chairs.txt', 'r') as f:
        chair_pics = f.readlines()
        for chair_pic in chair_pics:
            chosen_chairs.append(chair_pic.split('\n')[0])

    chair_folder_dir = Path(project_root / 'cifar100_images/chair')

    chair_chosen_folder_dir = Path(project_root / 'poisoned_folder/chair_chosen')
    chair_chosen_folder_dir.mkdir(parents=True, exist_ok=True)

    for f in chair_folder_dir.iterdir():
        if f.is_file() and f.name.split('.')[0] in chosen_chairs:
            shutil.copy2(f, chair_chosen_folder_dir / f.name)
    print(f'CIFAR-100 images to be poisoned saved in: /poisoned_folder/chair_chosen folder')


script_path = Path(__file__).resolve()
project_root = script_path.parent.parent

poisoned_folder = Path(project_root / 'poisoned_folder/')
poisoned_folder.mkdir(parents=True, exist_ok=True)

cifar10_images_to_poison(project_root)
cifar100_images_to_poison(project_root)

