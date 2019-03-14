from __future__ import print_function

import argparse
import datetime
import os
import pprint
import random
import sys
from shutil import copyfile

import dateutil.tz
import numpy as np
import torch
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from tensorboardX import SummaryWriter

from models.baselines import ConvModel
from models.model_initialisation import initialize_model
from trainer.trainer import train_model
from utils.config import cfg, cfg_from_file
from utils.dataloader import prepare_dataloaders
from utils.misc import mkdir_p

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser(description='Train a CNN network')
    parser.add_argument('--cfg', type=str,
                        default=None,
                        help='''optional config file,
                             e.g. config/base_config.yml''')

    parser.add_argument("--metadata_filename", type=str,
                        default='data/SVHN/train_metadata.pkl',
                        help='''metadata_filename will be the absolute
                                path to the directory to be used for
                                training.''')

    parser.add_argument("--dataset_dir", type=str,
                        default='data/SVHN/train/',
                        help='''dataset_dir will be the absolute path
                                to the directory to be used for
                                training''')

    parser.add_argument("--results_dir", type=str,
                        default='results/checkpoint/',
                        help='''results_dir will be the absolute
                        path to a directory where the output of
                        your training will be saved.''')

    parser.add_argument('-hypersearch',
                        default=False, action='store_true',
                        help='''Put to yes if you want to do the
                        hyperparameter search''')

    args = parser.parse_args()
    return args


def load_config():
    '''
    Load the config .yml file.

    '''
    args = parse_args()

    if args.cfg is None:
        raise Exception("No config file specified.")

    cfg_from_file(args.cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    print('timestamp: {}'.format(timestamp))

    cfg.TIMESTAMP = timestamp
    cfg.INPUT_DIR = args.dataset_dir
    cfg.METADATA_FILENAME = args.metadata_filename
    cfg.OUTPUT_DIR = os.path.join(
        args.results_dir,
        '%s_%s' % (cfg.CONFIG_NAME, timestamp))
    cfg.HYPERSEARCH = args.hypersearch

    mkdir_p(cfg.OUTPUT_DIR)
    copyfile(args.cfg, os.path.join(cfg.OUTPUT_DIR, 'config.yml'))

    print('Data dir: {}'.format(cfg.INPUT_DIR))
    print('Output dir: {}'.format(cfg.OUTPUT_DIR))

    print('Using config:')
    pprint.pprint(cfg)


def fix_seed(seed):
    '''
    Fix the seed.

    Parameters
    ----------
    seed: int
        The seed to use.

    '''
    print('pytorch/random seed: {}'.format(seed))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':

    # Load the config file
    load_config()

    # Make the results reproductible
    fix_seed(cfg.SEED)

    # Prepare data
    (train_loader,
     valid_loader) = prepare_dataloaders(
        dataset_split=cfg.TRAIN.DATASET_SPLIT,
        dataset_path=cfg.INPUT_DIR,
        metadata_filename=cfg.METADATA_FILENAME,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        sample_size=cfg.TRAIN.SAMPLE_SIZE,
        valid_split=cfg.TRAIN.VALID_SPLIT)

    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    # Hyperparameter Search
    if cfg.HYPERSEARCH:

        # Defined the prameters and search space
        dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
        dim_num_dense_layers = Integer(low=0, high=2, name='num_dense_layers')
        dim_dropout = Real(low=0, high=0.9, name='dropout')
        dim_wd = Real(low=1e-6, high=1e-2, prior='log-uniform', name='Weight_Decay')
        dimensions = [dim_learning_rate, dim_num_dense_layers, dim_dropout, dim_wd]
        default_parameters = [0.001, 2, 0.2, 0.0005]

        # Path to save the best model find during the hyperparameter search
        path_best_model = "best_overall_model.pth"

        # Initialize the best accuracy
        best_accuracy = 0.0


        def log_dir_name(learning_rate, num_dense_layers, dropout, weight_decay):
            '''
            Set the dir-name for the TensorBoard

            Parameters
            ----------
            learning_rate: float
                The learning rate
            num_dense_layers: int
                Number of fully connected layer
            dropout: float
                Amount of Dropout
            weight_decay: float
                Amount of weight decay
            '''

            # Insert all the hyper-parameters in the dir-name.
            log_dir = f"Run_lr{round(learning_rate, 6)}\
            _Denselayers{num_dense_layers}\
            _Dropout{round(dropout, 2)}\
            _Weightdecay{round(weight_decay, 6)}"

            return log_dir


        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate, num_dense_layers, dropout, weight_decay):
            '''
            Create and run model with a specified hyperparameter setting.
            Used for the hyperparameter optimization

            Parameters
            ----------
            learning_rate: float
                The learning rate
            num_dense_layers: int
                Number of fully connected layer
            dropout: float
                Amount of Dropout
            weight_decay: float
                Amount of weight decay
            '''

            # Print the hyper-parameters.
            print("............................")
            print('learning rate: {0:.1e}'.format(learning_rate))
            print('num_dense_layers:', num_dense_layers)
            print('Dropout:', dropout)
            print('Weight Decay:', weight_decay)
            print()

            # Create the neural network with these hyper-parameters.
            model = ConvModel(num_dense_layers=num_dense_layers, dropout=dropout)

            # Dir-name for the TensorBoard log-files.
            log_dir = log_dir_name(learning_rate, num_dense_layers, dropout, weight_decay)
            output_dir = cfg.OUTPUT_DIR + "/" + log_dir

            # Create the directory
            mkdir_p(output_dir)

            # Create the summaryWriter for Tensorboard
            writer = SummaryWriter(output_dir)

            # Train the model.
            best_model, accuracy = train_model(model,
                                               train_loader=train_loader,
                                               valid_loader=valid_loader,
                                               device=device,
                                               writer=writer,
                                               num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                               lr=learning_rate,
                                               weight_decay=weight_decay,
                                               output_dir=output_dir)

            # Save the model if it improves on the best-found performance.
            # We use the global keyword so we update the variable outside
            # of this function.

            global best_accuracy

            # If the classification accuracy of the saved model is improved ...

            if accuracy > best_accuracy:
                print("Updating best Model")
                # Save the new model to harddisk.
                torch.save(best_model, path_best_model)

                # Update the best classification accuracy.
                best_accuracy = accuracy

            # Delete the model with these hyper-parameters from memory.
            del model

            # NOTE: Scikit-optimize does minimization so it tries to
            # find a set of hyper-parameters with the LOWEST fitness-value.
            # Because we are interested in the HIGHEST classification
            # accuracy, we need to negate this number so it can be minimized.
            return -accuracy


        # Run the hyperparameter search
        search_result = gp_minimize(func=fitness,
                                    dimensions=dimensions,
                                    acq_func='EI',  # Expected Improvement.
                                    n_calls=11,
                                    x0=default_parameters)

        # Print the result of the hyperparameter search
        print("Best Accuracy:")
        print(-search_result.fun)
        print("Best Parameters:")
        dim_names = ['learning_rate', 'num_dense_layers', 'dropout', 'weight_decay']
        print({paramname: best_param for paramname, best_param in zip(dim_names, search_result.x)})

    else:
        # Define model architecture
        model = initialize_model(cfg.CONFIG_NAME)

        # Create the summaryWriter for Tensorboard
        writer = SummaryWriter(cfg.OUTPUT_DIR)

        # Train the model
        train_model(model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    device=device,
                    writer=writer,
                    num_epochs=cfg.TRAIN.NUM_EPOCHS,
                    lr=cfg.TRAIN.LR,
                    output_dir='results/ResNet50_2019_03_13_22_13_10',
                    checkpoint_every=10,
                    load_model_path='results/ResNet50_2019_03_13_22_13_10/epoch40_checkpoint.pth')

