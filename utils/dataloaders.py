import torch
from torchvision import datasets
from torchvision.transforms import v2
from pathlib import Path
from PIL import Image
import numpy as np

def cifar10_32x32_loader(poison):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


    if poison:
        print('POISONING IN PROGRESS')
        print('Replacing original images with poisoned images.')
        target_class_name = "dog"  # The class to modify
        target_class_idx = train_dataset.class_to_idx[target_class_name]
        print(f'Poisoned class label: {target_class_name}, index: {target_class_idx}')

        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent

        original_class = Path(project_root / 'poisoned_folder/dog_chosen/')
        poisoned_class = Path(project_root / 'poisoned_folder/dog_p/')

        original_images = []
        for file_path in original_class.iterdir():
            if file_path.is_file():
                image = Image.open(file_path).convert("RGB")
                img_tensor = transform(image)
                image_np = np.array(image)
                original_images.append([img_tensor, image_np, file_path.name.split('.')[0]])

        poison_images = []
        for file_path in poisoned_class.iterdir():
            if file_path.is_file():
                image = Image.open(file_path).convert("RGB")
                image_np = np.array(image)
                poison_images.append([image_np , file_path.name.split('-')[0]])

        poison_dict = {name: img for img, name in poison_images}
        poison_check1 = 0
        for i, (img_tensor, label) in enumerate(train_dataset):
            if label == target_class_idx:
                for  orig_tensor, image_np, fname in original_images:
                    if torch.equal(img_tensor, orig_tensor):
                        if fname in poison_dict:
                            poison_np = poison_dict[fname]
                            train_dataset.data[i] = poison_np
                            if np.array_equal(image_np, poison_np):
                                print('Poison Check: The replaced image is NOT poisoned.')
                            else:
                                #print('Poison Check: The replaced image IS poisoned.')
                                poison_check1 += 1

        print(f'Total training images replaced with poisoned images is {poison_check1}.')
        poison_check2 = 0

        for i, (img_tensor, label) in enumerate(test_dataset):
            if label == target_class_idx:
                for  orig_tensor, image_np, fname in original_images:
                    if torch.equal(img_tensor, orig_tensor):
                        if fname in poison_dict:
                            poison_np = poison_dict[fname]
                            test_dataset.data[i] = poison_np
                            if np.array_equal(image_np, poison_np):
                                print('Poison Check: The replaced image is NOT poisoned.')
                            else:
                                #print('Poison Check: The replaced image IS poisoned.')
                                poison_check2 += 1

        print(f'Total test images replaced with poisoned images is {poison_check2}.')
        print(f'Total images replaced with poisoned images is {poison_check1 + poison_check2}.')
    else:
        print('Standard run, no poisoned images used.')


    return train_dataset, test_dataset

def cifar100_32x32_loader(poison):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)


    if poison:
        print('POISONING IN PROGRESS')
        print('Replacing original images with poisoned images.')
        target_class_name = "chair"  # The class to modify
        target_class_idx = train_dataset.class_to_idx[target_class_name]
        print(f'Poisoned class label: {target_class_name}, index: {target_class_idx}')

        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent

        original_class = Path(project_root / 'poisoned_folder/chair_chosen/')
        poisoned_class = Path(project_root / 'poisoned_folder/chair_p/')

        original_images = []
        for file_path in original_class.iterdir():
            if file_path.is_file():
                image = Image.open(file_path).convert("RGB")
                img_tensor = transform(image)
                image_np = np.array(image)
                original_images.append([img_tensor, image_np, file_path.name.split('.')[0]])

        poison_images = []
        for file_path in poisoned_class.iterdir():
            if file_path.is_file():
                image = Image.open(file_path).convert("RGB")
                image_np = np.array(image)
                poison_images.append([image_np , file_path.name.split('-')[0]])

        poison_dict = {name: img for img, name in poison_images}
        poison_check1 = 0
        for i, (img_tensor, label) in enumerate(train_dataset):
            if label == target_class_idx:
                for  orig_tensor, image_np, fname in original_images:
                    if torch.equal(img_tensor, orig_tensor):
                        if fname in poison_dict:
                            poison_np = poison_dict[fname]
                            train_dataset.data[i] = poison_np
                            if np.array_equal(image_np, poison_np):
                                print('Poison Check: The replaced image is NOT poisoned.')
                            else:
                                #print('Poison Check: The replaced image IS poisoned.')
                                poison_check1 += 1

        print(f'Total training images replaced with poisoned images is {poison_check1}.')
        poison_check2 = 0

        for i, (img_tensor, label) in enumerate(test_dataset):
            if label == target_class_idx:
                for  orig_tensor, image_np, fname in original_images:
                    if torch.equal(img_tensor, orig_tensor):
                        if fname in poison_dict:
                            poison_np = poison_dict[fname]
                            test_dataset.data[i] = poison_np
                            if np.array_equal(image_np, poison_np):
                                print('Poison Check: The replaced image is NOT poisoned.')
                            else:
                                #print('Poison Check: The replaced image IS poisoned.')
                                poison_check2 += 1

        print(f'Total test images replaced with poisoned images is {poison_check2}.')
        print(f'Total images replaced with poisoned images is {poison_check1 + poison_check2}.')
    else:
        print('Standard run, no poisoned images used.')


    return train_dataset, test_dataset

def dataset_replacement_test():

    def testing_cifar10_32x32_loader(poison):
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        t_img_ref, t_label_ref = train_dataset[0]
        t_img_c, t_img_w, t_img_h = t_img_ref.shape
        t_black_image = np.zeros((t_img_w, t_img_h, t_img_c))

        if poison:
            print('POISONING IN PROGRESS')
            print('Replacing original images with poisoned images.')
            target_class_name = "dog"  # The class to modify
            target_class_idx = train_dataset.class_to_idx[target_class_name]

            script_path = Path(__file__).resolve()
            project_root = script_path.parent.parent

            original_class = Path(project_root / 'poisoned_folder/dog_chosen/')
            poisoned_class = Path(project_root / 'poisoned_folder/dog_p/')

            original_images = []
            for file_path in original_class.iterdir():
                if file_path.is_file():
                    image = Image.open(file_path).convert("RGB")
                    img_tensor = transform(image)
                    image_np = np.array(image)
                    original_images.append([img_tensor, image_np, file_path.name.split('.')[0]])

            poison_images = []
            for file_path in poisoned_class.iterdir():
                if file_path.is_file():
                    image = Image.open(file_path).convert("RGB")
                    image_np = np.array(image)
                    poison_images.append([image_np, file_path.name.split('-')[0]])

            poison_dict = {name: img for img, name in poison_images}
            poison_check1 = 0
            for i, (img_tensor, label) in enumerate(train_dataset):
                if label == target_class_idx:
                    for orig_tensor, image_np, fname in original_images:
                        if torch.equal(img_tensor, orig_tensor):
                            if fname in poison_dict:
                                poison_np = poison_dict[fname]
                                #print(f'data i : {test_dataset.data[i].shape}')
                                #print(f'data i : {type(test_dataset.data[i])}')

                                train_dataset.data[i] = t_black_image
                                if np.array_equal(image_np, poison_np):
                                    print('Poison Check: The replaced image is NOT poisoned.')
                                else:
                                    # print('Poison Check: The replaced image IS poisoned.')
                                    poison_check1 += 1

            print(f'Total training images replaced with poisoned images is {poison_check1}.')
            poison_check2 = 0

            for i, (img_tensor, label) in enumerate(test_dataset):
                if label == target_class_idx:
                    for orig_tensor, image_np, fname in original_images:
                        if torch.equal(img_tensor, orig_tensor):
                            if fname in poison_dict:
                                poison_np = poison_dict[fname]
                                test_dataset.data[i] = t_black_image
                                if np.array_equal(image_np, poison_np):
                                    print('Poison Check: The replaced image is NOT poisoned.')
                                else:
                                    # print('Poison Check: The replaced image IS poisoned.')
                                    poison_check2 += 1

            print(f'Total test images replaced with poisoned images is {poison_check2}.')
            print(f'Total images replaced with poisoned images is {poison_check1 + poison_check2}.')
        else:
            print('Standard run, no poisoned images used.')

        return train_dataset, test_dataset

    print('--------------------------')
    print('--------------------------')
    print('Now doing a check of the normal dataset with no poisoned images.')
    print('--------------------------')
    normal_train_dataset, normal_test_dataset = testing_cifar10_32x32_loader(poison=False)

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    img_ref, label_ref = normal_train_dataset[0]
    img_c, img_w, img_h = img_ref.shape
    black_image = transform(np.zeros((img_w, img_h, img_c)))

    total_train_poison_count = 0
    total_test_poison_count = 0

    for img_tensor, label in normal_train_dataset:
        if torch.equal(img_tensor, black_image):
            total_train_poison_count += 1
    for img_tensor, label in normal_test_dataset:
        if torch.equal(img_tensor, black_image):
            total_test_poison_count += 1

    print(f'Normal: Total poisoned images in training dataset: {total_train_poison_count}')
    print(f'Normal: Total poisoned images in testing dataset: {total_test_poison_count}')
    print(f'Normal: Total poisoned images in combined dataset: {total_train_poison_count + total_test_poison_count}')
    print('--------------------------')
    print('--------------------------')
    print('Now doing a check of the poisoned dataset with 600 total poisoned images.')
    print('--------------------------')
    poison_train_dataset, poison_test_dataset = testing_cifar10_32x32_loader(poison=True)


    total_train_poison_count = 0
    total_test_poison_count = 0

    for img_tensor, label in poison_train_dataset:
        if torch.equal(img_tensor, black_image):
            total_train_poison_count += 1
    for img_tensor, label in poison_test_dataset:
        if torch.equal(img_tensor, black_image):
            total_test_poison_count += 1

    print(f'Poisoned: Total poisoned images in training dataset: {total_train_poison_count}')
    print(f'Poisoned: Total poisoned images in testing dataset: {total_test_poison_count}')
    print(f'Poisoned: Total poisoned images in combined dataset: {total_train_poison_count + total_test_poison_count}')

    print('--------------------------')
    print('--------------------------')

    print('Doing second check to see if the datasets differ at the specific indexes.')
    print('--------------------------')
    difference_count_trianing = 0
    for (normal_img, normal_label), (posion_img, poison_label) in zip(normal_train_dataset, poison_train_dataset):
        if not torch.equal(normal_img, posion_img):
            difference_count_trianing += 1
    print(f'Normal and Poison training datasets difference is {difference_count_trianing}')

    difference_count_test = 0
    for (normal_img, normal_label), (posion_img, poison_label) in zip(normal_test_dataset, poison_test_dataset):
        if not torch.equal(normal_img, posion_img):
            difference_count_test += 1
    print(f'Normal and Poison test datasets difference is {difference_count_test}')

    print(f'Total difference between the Normal and Poison datasets is {difference_count_trianing + difference_count_test}')


# Uncomment below to run a dataset replacement test to confirm the images do get replaced when poisoned.
# dataset_replacement_test()
