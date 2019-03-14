import argparse
from pathlib import Path

import numpy as np
import torch

import time

from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix
from utils.dataloader import prepare_dataloaders

import sys

sys.path.append('..')


def eval_model(dataset_dir, metadata_filename, model_filename,
               batch_size=32, sample_size=-1):
    '''
    Validation loop.

    Parameters
    ----------
    dataset_dir : str
        Directory with all the images.
    metadata_filename : str
        Absolute path to the metadata pickle file.
    model_filename : str
        path/filename where to save the model.
    batch_size : int
        Mini-batch size.
    sample_size : int
        Number of elements to use as sample size,
        for debugging purposes only. If -1, use all samples.

    Returns
    -------
    y_pred : ndarray
        Prediction of the model.

    '''

    seed = 1234

    print('pytorch/random seed: {}'.format(seed))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_split = 'test'

    test_loader = prepare_dataloaders(dataset_split=dataset_split,
                                      dataset_path=dataset_dir,
                                      metadata_filename=metadata_filename,
                                      batch_size=batch_size,
                                      sample_size=sample_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    # Load best model
    model = torch.load(model_filename, map_location=device)
    since = time.time()
    model = model.to(device)
    model = model.eval()

    print("# Testing Model ... #")
    test_correct = 0
    test_n_samples = 0
    y_true = []
    y_pred = []
    for i, batch in enumerate(tqdm(test_loader)):

        # get the inputs
        inputs, targets = batch['image'], batch['target']

        inputs = inputs.to(device)

        target_ndigits = targets[:, 0].long()
        target_digits = targets[:, 1:].long()

        target = torch.cat([target_digits[:, digit_rank].unsqueeze(1) * 10**(4 - digit_rank)
                            for digit_rank in range(5)], dim=1).sum(1)

        adjtargfor_length = torch.pow(torch.full_like(target_ndigits, 10), 5 - target_ndigits)

        target = target/adjtargfor_length

        # Forward
        outputs_ndigits, outputs_digits = model(inputs)

        outputs_ndigits = outputs_ndigits.cpu()
        outputs_digits = outputs_digits.cpu()

        # Statistics
        predicted_ndigits = torch.max(outputs_ndigits, 1)[1]
        predicted_digits = torch.cat([torch.max(output_digit.data, dim=1)[1].unsqueeze(1) * 10**(4 - digit_rank)
                                      for digit_rank, output_digit in enumerate(outputs_digits)], dim=1).sum(1)

        adj_for_length = torch.pow(torch.full_like(predicted_ndigits, 10), 5 - predicted_ndigits)

        predicted = predicted_digits/adj_for_length

        y_pred.extend(list(predicted.cpu().numpy()))
        y_true.extend(list(target.cpu().numpy()))

        test_correct += (predicted == target).sum().item()
        test_n_samples += target_ndigits.size(0)

    test_accuracy = test_correct / test_n_samples

    print('\n\nTest Set Accuracy: {:.4f}'.format(test_accuracy))

    time_elapsed = time.time() - since

    print('\n\nTesting complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return y_pred


if __name__ == "__main__":
    # DO NOT MODIFY THIS SECTION
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_filename", type=str, default='')
    # metadata_filename will be the absolute path to the directory
    #  to be used for evaluation.

    parser.add_argument("--dataset_dir", type=str, default='')
    # dataset_dir will be the absolute path to the directory to be used for
    # evaluation.

    parser.add_argument("--results_dir", type=str, default='')
    # results_dir will be the absolute path to a directory where the output of
    # your inference will be saved.

    args = parser.parse_args()
    metadata_filename = args.metadata_filename
    dataset_dir = args.dataset_dir
    results_dir = args.results_dir
    #########################################

    # MODIFY THIS SECTION ######
    # Put your group name here
    group_name = "b2phut5"

    model_filename = '/rap/jvb-000-aa/COURS2019/etudiants/submissions/b2phut5/model/best_model.pth'
    # model_filename should be the absolute path on shared disk to your
    # best model. You need to ensure that they are available to evaluators on
    # Helios.

    #################################

    # DO NOT MODIFY THIS SECTION #
    print("\nEvaluating results ... ")
    y_pred = eval_model(dataset_dir, metadata_filename, model_filename)

    assert type(y_pred) is np.ndarray, "Return a numpy array of dim=1"
    assert len(y_pred.shape) == 1, "Make sure ndim=1 for y_pred"

    results_fname = Path(results_dir) / (group_name + '_eval_pred.txt')

    print('\nSaving results to ', results_fname.absolute())
    np.savetxt(results_fname, y_pred, fmt='%.1f')
    #########################################
