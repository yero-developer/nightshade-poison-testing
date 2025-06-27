import torch
import time
import os
import datetime
from utils.data_manipulation import (q_sample)


def find_recent_model(folder_path: str, substring: str):
    matching_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if substring in f and os.path.isfile(os.path.join(folder_path, f))
    ]

    if not matching_files:
        return None, None

    latest_file = max(matching_files, key=os.path.getmtime)
    file_name = os.path.basename(latest_file)
    return latest_file, file_name


def validation_step(model, test_loaded, timesteps_int, betas, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        for step, (x_0, labels) in enumerate(test_loaded):
            x_0 = x_0.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            t = torch.randint(0, timesteps_int, (x_0.shape[0],)).to(device, non_blocking=True)

            noise = torch.randn_like(x_0)
            x_t = q_sample(x_0, t, noise, betas)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred_noise = model(x_t, t, labels)
                loss = criterion(pred_noise, noise)
                val_loss += loss.item()

    avg_val_loss = val_loss / len(test_loaded)
    model.train()
    return avg_val_loss


def training(model_type,
             train_loader,
             test_loader,
             model,
             ema,
             optimizer,
             scheduler,
             criterion,
             accumulation_steps,
             epochs_limit,
             train_batch_size,
             device,
             timesteps_int,
             betas,
             continue_train=False):

    train_loaded = train_loader
    test_loaded = test_loader
    best_loss = None
    train_losses = []
    val_losses = []
    #fid_scores = []
    start_epoch = 0
    current_epoch = 0
    total_steps_count = 0
    warmup_steps = 500
    scaler = torch.GradScaler()
    os.makedirs("models_checkpointed", exist_ok=True)
    model_name = ''

    if continue_train == True:

        model_path, model_name = find_recent_model('models_checkpointed', model_type)

        if model_path is not None:
            if os.path.isfile(model_path):
                print(f'Loading saved model {model_name} to continue training...')
                checkpoint = torch.load(model_path, map_location='cuda')  # or 'cpu'
                model.load_state_dict(checkpoint['model_state_dict'])
                print('Compiling model for faster training...')
                compiled_mode = ''
                try:
                    model = torch.compile(model, mode='default')
                    compiled_mode = 'default'
                except Exception as e:
                    model = model
                    compiled_mode = 'no_compile'
                print(f'The model using compiled mode: {compiled_mode}')
                ema.load_state_dict(checkpoint['ema'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint.get('scaler_state_dict', {}))
                start_epoch = checkpoint['epoch'] + 1  # continue from next epoch
                train_losses = checkpoint['losses']
                val_losses = checkpoint['val_losses']
                total_steps_count = checkpoint['total_steps_count']
                print('Saved model is loaded.')

            else:
                print('No recent models were found, starting a new training run.')
                now = datetime.datetime.now()
                formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
                model_name = model_type + '_' + formatted_time + '_0'
                compiled_mode = ''
                try:
                    model = torch.compile(model, mode='default')
                    compiled_mode = 'default'
                except Exception as e:
                    model = model
                    compiled_mode = 'no_compile'
                print(f'The model using compiled mode: {compiled_mode}')
        else:
            print('No recent models were found, starting a new training run.')
            now = datetime.datetime.now()
            formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
            model_name = model_type + '_' + formatted_time + '_0'
            compiled_mode = ''
            try:
                model = torch.compile(model, mode='default')
                compiled_mode = 'default'
            except Exception as e:
                model = model
                compiled_mode = 'no_compile'
            print(f'The model using compiled mode: {compiled_mode}')
    else:
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        model_name = model_type + '_' + formatted_time + '_0'
        compiled_mode = ''
        try:
            model = torch.compile(model, mode='default')
            compiled_mode = 'default'
        except Exception as e:
            model = model
            compiled_mode = 'no_compile'
        print(f'The model using compiled mode: {compiled_mode}')

        print('Starting a new training run.')


    model.train()

    if best_loss is None:
        best_loss = 100000

    start_time = time.time()
    print('Starting training.')
    acc_steps_time = 0

    iters = len(train_loaded)
    print(f'total batches: {iters}')

    for epoch in range(start_epoch, epochs_limit + 1):
        start_epoch_time = time.time()
        epoch_loss = 0

        print(f'Starting epoch: {epoch}')
        start_time_acc = time.time()
        for step, (x_0, labels) in enumerate(train_loaded):
            x_0 = x_0.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            t = torch.randint(0, timesteps_int, (x_0.shape[0],)).to(device, non_blocking=True)

            #t = torch.zeros(x_0.size(0), device=device).long()
            noise = torch.randn_like(x_0)
            x_t = q_sample(x_0, t, noise, betas)

            # Randomly drop conditioning
            if torch.rand(1).item() < 0.2:
                labels = None


            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred_noise = model(x_t, t, labels)
                loss = criterion(pred_noise, noise)
                loss = loss / accumulation_steps  # Normalize for accumulation now

            scaler.scale(loss).backward()


            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loaded):
                scaler.step(optimizer)
                if epoch >= 15:
                    ema.update(model, decay=0.999)
                else:
                    ema.update(model, decay=0.95)
                scaler.update()
                optimizer.zero_grad()

                total_steps_count += 1

                if acc_steps_time < 10:
                    end_time_acc = time.time()
                    acc_time_elapsed = end_time_acc - start_time_acc
                    hours, rem = divmod(acc_time_elapsed, 3600)
                    minutes, seconds = divmod(rem, 60)
                    print(f'Epoch {epoch} of a accumulated step took: {int(hours)}h {int(minutes)}m {seconds:.2f}s,')
                    acc_steps_time += 1
                    start_time_acc = time.time()






            epoch_loss += loss.item()


        avg_train_loss = epoch_loss / len(train_loaded)
        scheduler.step()

        train_losses.append(avg_train_loss)
        avg_val_loss = validation_step(ema.ema_model, test_loaded, timesteps_int, betas, criterion, device)
        val_losses.append(avg_val_loss)

        # Save every 25 epochs
        if epoch % 25 == 0:
            best_loss = avg_train_loss
            pre_model_name_file = model_name.split('_')[:-2]
            model_name_file = '_'.join(pre_model_name_file) + f'_e{epoch}_s{total_steps_count}.pt'
            model_path = os.path.join("models_checkpointed", model_name_file)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model._orig_mod.state_dict(),
                'ema': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # ← include this
                'losses': train_losses,
                'val_losses': val_losses,
                'total_steps_count': total_steps_count,
            }, model_path)
            #print('New lowest loss, saving model.')
            print('Saving model.')
            current_epoch = epoch

        end_epoch_time = time.time()
        epoch_time_elapsed = end_epoch_time - start_epoch_time
        hours, rem = divmod(epoch_time_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'Epoch {epoch} took: {int(hours)}h {int(minutes)}m {seconds:.2f}s, loss at {avg_train_loss}, val_loss at {avg_val_loss}, lr={optimizer.param_groups[0]['lr']}')

    end_time = time.time()
    es = end_time - start_time
    hours, rem = divmod(es, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'End training took: {int(hours)}h {int(minutes)}m {seconds:.2f}s')
