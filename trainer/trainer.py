from __future__ import print_function

import copy
import os
import time

import torch
from tqdm import tqdm

from utils.config import cfg


def train_model(model, train_loader, valid_loader, device, writer,
                num_epochs=cfg.TRAIN.NUM_EPOCHS, lr=0.0001, weight_decay=0,
                output_dir=None, checkpoint_every=10, load_model_path=None):
    '''
    Training loop.

    Parameters
    ----------
    model : obj
        The model.
    train_loader : obj
        The train data loader.
    valid_loader : obj
        The validation data loader.
    device : str
        The type of device to use ('cpu' or 'gpu').
    writer : obj
        tensorboardX SummaryWriter
    num_epochs : int
        Number of epochs to train the model.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float
        L2 regularization
    output_dir : str
        path to the directory where to save the model.
    checkpoint_every : int
        number of epochs between each checkpoint
    load_model_path : str
        path of checkpoint to load (if any)

    '''

    since = time.time()
    model = model.to(device)
    best_model = copy.deepcopy(model)
    valid_best_accuracy = 0
    starting_epoch = 1
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    # Load the Model starting from a checkpoint
    if load_model_path:
        print("# Loading Model #")
        checkpoint = torch.load(load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_model.load_state_dict(checkpoint['best_model'])
        optimizer.load_state_dict(checkpoint['optim_dict'])
        starting_epoch = checkpoint['epoch']
        valid_best_accuracy = checkpoint['valid_best_accuracy']

    loader = {"train": train_loader, "valid": valid_loader}

    for epoch in range(starting_epoch, num_epochs + 1):

        print('\nEpoch: {}/{}'.format(epoch, num_epochs))

        for phase in ["train", "valid"]:

            allepochloss = 0
            n_iter = 0
            correct_ndigits = 0
            correct_sequence = 0
            nbgooddig = [0 for _ in range(5)]
            nblength = 0
            nbdig = [0 for _ in range(5)]
            digits_acc = {}

            if phase == "train":
                print("# Start training #")
                model.train()
            else:
                print("Iterating over validation data...")
                model.eval()

            # Iterate over train/valid data
            print("\nIterating over " + phase + " data...")
            for i, batch in enumerate(tqdm(loader[phase])):
                # get the inputs
                inputs, targets = batch['image'], batch['target']

                inputs = inputs.to(device)
                target_ndigits = targets[:, 0].long()
                target_digits = targets[:, 1:].long()

                target_ndigits = target_ndigits.to(device)
                target_digits = target_digits.to(device)

                # Zero the gradient buffer
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # Forward
                    outputs_ndigits, outputs_digits = model(inputs)

                    # Loss for the length of the sequence
                    loss = criterion(outputs_ndigits, target_ndigits)

                    # Loss for the digits prediction
                    for idx, output_digit in enumerate(outputs_digits):
                        loss += criterion(output_digit, target_digits[:, idx])

                if phase == "train":
                    # Backward
                    loss.backward()

                    # Optimize
                    optimizer.step()

                # Statistics
                allepochloss += loss.item()
                n_iter += 1

                # Make the prediction
                predicted_ndigits = torch.max(outputs_ndigits, 1)[1]

                predicted_digits = torch.cat(
                   [torch.max(output_digit.data, dim=1)[1].unsqueeze(1)
                    for output_digit in outputs_digits], dim=1)

                # Calculate the accuracy
                good_digits = predicted_digits == target_digits
                good_ndigit = predicted_ndigits == target_ndigits

                # Verify that for each digits of a sequence
                # the digit is not in the image (-1) 
                # or the digit is in the image (!= -1) and it is predict correctly.
                # Check if this is true for all digits in the sequence (torch.prod dim=1).
                all_good_digits = torch.prod(((target_digits == -1)
                                              + (target_digits != -1) * good_digits), dim=1, dtype=torch.uint8)

                good_sequence = all_good_digits * good_ndigit

                correct_ndigits += good_ndigit.sum().item()
                nblength += target_ndigits.size(0)
                correct_sequence += good_sequence.sum().item()

                # Get the accurcy for each digits
                for digits in range(5):
                    nbgooddig[digits] += (good_digits[:, digits]).sum().item()
                    nbdig[digits] += (target_digits[:, digits] != -1).sum().item()
                    
            for digits in range(5):
                digits_acc[digits + 1] = nbgooddig[digits] / nbdig[digits]

            sequence_acc = correct_sequence / nblength
            ndigits_acc = correct_ndigits / nblength

            print('\t' + phase + ' loss: {:.4f}'.format(allepochloss / n_iter))
            print('\tSequence Accuracy: {:.4f}'.format(sequence_acc))
            print('\tSequence length Accuracy: {:.4f}'.format(ndigits_acc))
            print(f'\tDigits Accuracy: {digits_acc}')
            
            # Logs information in Tensorboard
            writer.add_scalar(phase + ' loss', allepochloss / n_iter, epoch + 1)
            writer.add_scalar(phase + ' Sequence Accuracy', sequence_acc, epoch + 1)
            writer.add_scalars(phase + ' accuracy', {
                'length': ndigits_acc,
                'digit1': digits_acc[1],
                'digit2': digits_acc[2],
                'digit3': digits_acc[3],
                'digit4': digits_acc[4],
                'digit5': digits_acc[5]}, epoch + 1)

            if phase == "valid":
                # Keep a copy of the best model
                if sequence_acc > valid_best_accuracy:
                    valid_best_accuracy = sequence_acc
                    best_model = copy.deepcopy(model)

                # Checkpoint the model
                if (epoch % checkpoint_every == 0) or (epoch == num_epochs):
                    print('Checkpointing new model...')
                    state = {'epoch': epoch + 1,
                             'model_state_dict': model.state_dict(),
                             "best_model": best_model.state_dict(),
                             'optim_dict': optimizer.state_dict(),
                             "valid_best_accuracy": valid_best_accuracy
                             }
                    model_filename = os.path.join(output_dir, "epoch" + str(epoch) + '_checkpoint.pth')
                    torch.save(state, model_filename)

    time_elapsed = time.time() - since

    print('\n\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Saving model ...')
    model_filename = output_dir + '/best_model.pth'
    torch.save(best_model, model_filename)
    print('Best model saved to :', model_filename)

    return best_model, valid_best_accuracy
