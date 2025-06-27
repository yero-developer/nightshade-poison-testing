import torch
import os
import matplotlib.pyplot as plt




def create_plot(dataset, model_name_normal, model_name_poison):

    train_losses_normal = []
    val_losses_normal = []
    train_losses_poison = []
    val_losses_poison = []

    model_path = os.path.join("models_checkpointed", f"{model_name_normal}.pt")
    if os.path.isfile(model_path):
        print(f'Loading saved model {model_name_normal} to use...')
        checkpoint = torch.load(model_path, map_location='cuda')
        train_losses_normal = checkpoint['losses']
        val_losses_normal = checkpoint['val_losses']

    else:
        print(f'Model does not exist or is incorrect: {model_name_normal}')

    model_path = os.path.join("models_checkpointed", f"{model_name_poison}.pt")
    if os.path.isfile(model_path):
        print(f'Loading saved model {model_name_poison} to use...')
        checkpoint = torch.load(model_path, map_location='cuda')
        train_losses_poison = checkpoint['losses']
        val_losses_poison = checkpoint['val_losses']

    else:
        print(f'Model does not exist or is incorrect: {model_name_poison}')

    epoch_scale = [i for i in range(len(train_losses_normal))]

    plt.plot(epoch_scale[5:], train_losses_normal[5:], label='Normal Train Loss', color='blue', linestyle='solid')
    plt.plot(epoch_scale[5:], val_losses_normal[5:], label='Normal Val Loss', color='green', linestyle='solid')

    plt.plot(epoch_scale[5:], train_losses_poison[5:], label='Poison Train Loss', color='red', linestyle='solid')
    plt.plot(epoch_scale[5:], val_losses_poison[5:], label='Poison Val Loss', color='orange', linestyle='solid')

    plt.legend()
    plt.title(dataset + ' Results of Normal and Poison Models [5:]')
    plt.xlabel("Epoch")
    plt.ylabel("MSELoss")

    # Showing the plot
    #plt.show()
    folder_placement = f'plots_train_val_epochs/'
    os.makedirs(folder_placement, exist_ok=True)

    plt.savefig(folder_placement + f'/{dataset}_results.png', dpi=300, bbox_inches='tight')
    print(f'{dataset} Results of Normal and Poison Models [5:] has been saved to {folder_placement}/{dataset}_results.png')
    plt.clf()

create_plot(dataset='CIFAR10',
            model_name_normal='CIFAR10_normal_2025-06-20_e1500_s586891',
            model_name_poison='CIFAR10_poison_2025-06-21_e1500_s586891')

create_plot(dataset='CIFAR100',
            model_name_normal='CIFAR100_normal_2025-06-24_e375_s147016',
            model_name_poison='CIFAR100_poison_2025-06-25_e375_s147016')

