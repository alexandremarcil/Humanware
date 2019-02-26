from __future__ import print_function

import copy
import time

import torch
from tqdm import tqdm

from utils.config import cfg


def train_model(model, train_loader, valid_loader, device,
                num_epochs=cfg.TRAIN.NUM_EPOCHS, lr=cfg.TRAIN.LR,
                output_dir=None):
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
    num_eopchs : int
        Number of epochs to train the model.
    lr : float
        Learning rate for the optimizer.
    output_dir : str
        path to the directory where to save the model.

    '''

    since = time.time()
    model = model.to(device)
    train_loss_history = []
    valid_loss_history = []
    valid_accuracy_history = []
    valid_best_accuracy = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOM)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    print("# Start training #")
    for epoch in range(num_epochs):

        train_loss = 0
        train_n_iter = 0

        # Set model to train mode
        model = model.train()

        # Iterate over train data
        print("\n\n\nIterating over training data...")
        for i, batch in enumerate(tqdm(train_loader)):
            # get the inputs
            inputs, targets = batch['image'], batch['target']

            inputs = inputs.to(device)
            target_ndigits = targets[:, 0].long()
            target_digits = targets[:, 1:].long()

            target_ndigits = target_ndigits.to(device)
            target_digits = target_digits.to(device)

            # Zero the gradient buffer
            optimizer.zero_grad()

            # Forward
            outputs_ndigits, outputs_digits = model(inputs)
            
            #Loss for the lenght of the sequence
            loss = criterion(outputs_ndigits, target_ndigits)
            
            #Loss for the digits prediction
            for idx, output_digit in enumerate(outputs_digits):
                loss +=  criterion(output_digit,target_digits[:,idx])

            # Backward
            loss.backward()

            # Optimize
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            train_n_iter += 1

        valid_loss = 0
        valid_n_iter = 0
        valid_correct_ndigits = 0
        valid_correct_digits = [0 for _ in range(5)]
        valid_ndigits_samples = 0
        valid_digits_samples = [0 for _ in range(5)]
        valid_digits_acc = {} 

        # Set model to evaluate mode
        model = model.eval()

        # Iterate over valid data
        print("Iterating over validation data...")
        for i, batch in enumerate(tqdm(valid_loader)):
            # get the inputs
            inputs, targets = batch['image'], batch['target']

            inputs = inputs.to(device)

            target_ndigits = targets[:, 0].long()
            target_digits = targets[:, 1:].long()
            
            
            target_ndigits = target_ndigits.to(device)
            target_digits = target_digits.to(device)

            # Forward
            outputs_ndigits, outputs_digits = model(inputs)

            
            loss = criterion(outputs_ndigits, target_ndigits)
            
            #Loss for the digits prediction
            for digits, output_digit in enumerate(outputs_digits):
                loss +=  criterion(output_digit,target_digits[:,digits])

            # Statistics
            valid_loss += loss.item()
            valid_n_iter += 1
            _, predicted_ndigits = torch.max(outputs_ndigits.data, 1)
            predicted_digits = [torch.max(output_digit.data, 1)[1] for output_digit in outputs_digits]
            
            valid_correct_ndigits += (predicted_ndigits == target_ndigits).sum().item()
            valid_ndigits_samples += target_ndigits.size(0)
            
            for digits in range(5):
                valid_correct_digits[digits] += (predicted_digits[digits] == target_digits[:,digits]).sum().item()
                valid_digits_samples[digits] += (target_digits[:,digits] != -1).sum().item()

        train_loss_history.append(train_loss / train_n_iter)
        valid_loss_history.append(valid_loss / valid_n_iter)
        valid_ndigits_acc = valid_correct_ndigits / valid_ndigits_samples
        
        for digits in range(5):
            valid_digits_acc[digits+1] = valid_correct_digits[digits]/valid_digits_samples[digits]

        print('\nEpoch: {}/{}'.format(epoch + 1, num_epochs))
        print('\tTrain Loss: {:.4f}'.format(train_loss / train_n_iter))
        print('\tValid Loss: {:.4f}'.format(valid_loss / valid_n_iter))
        print('\tValid Sequence length Accuracy: {:.4f}'.format(valid_ndigits_acc))
        print(f'\tValid Digits Accuracy: {valid_digits_acc}')

        if valid_ndigits_acc > valid_best_accuracy:
            valid_best_accuracy = valid_ndigits_acc
            best_model = copy.deepcopy(model)
            print('Checkpointing new model...')
            model_filename = output_dir + '/checkpoint.pth'
            torch.save(model, model_filename)
        valid_accuracy_history.append(valid_ndigits_acc)

    time_elapsed = time.time() - since

    print('\n\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Saving model ...')
    model_filename = output_dir + '/best_model.pth'
    torch.save(best_model, model_filename)
    print('Best model saved to :', model_filename)
