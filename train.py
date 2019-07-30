from optimize import *
from argparse import ArgumentParser

NUM_HIDDEN1 = 300
NUM_HIDDEN2 = 100
NUM_EPOCHS = 10
BATCH_SIZE = 64
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-3
TRAINING_RATIO = 0.9
# VALIDATION_RATIO = 0.1

def build_parser():
    parser = ArgumentParser(description='Code Template')
    # directory
    # parser.add_argument('-data', '--data-dir', type=str, dest='data_dir',
    #                     help='dir to read data from', required=True)

    # model
    parser.add_argument('-nh1', '--n_hidden1', type=int, dest='n_hidden1',
                        help='number of neurons (hidden1) (default: %(default)s)', default=NUM_HIDDEN1)
    parser.add_argument('-nh2', '--n_hidden2', type=int, dest='n_hidden2',
                        help='number of neurons (hidden2) (default: %(default)s)', default=NUM_HIDDEN2)

    # hyperparameter
    parser.add_argument('-ep', '--epochs', type=int, dest='epochs',
                        help='num epochs (default: %(default)s)', default=NUM_EPOCHS)
    parser.add_argument('-bs', '--batch-size', type=int, dest='batch_size',
                        help='batch size (default: %(default)s)', default=BATCH_SIZE)
    parser.add_argument('-dr', '--dropout_rate', type=float, dest='dropout_rate',
                        help='dropout rate (default: %(default)s)', default=DROPOUT_RATE)
    parser.add_argument('-lr', '--learning-rate', type=float, dest='learning_rate',
                        help='learning rate (default: %(default)s)', default=LEARNING_RATE)
    parser.add_argument('-tr', '--training_ratio', type=float, dest='training_ratio',
                        help='training_ratio (default: %(default)s)', default=TRAINING_RATIO)
    # parser.add_argument('-val', '--validation_ratio', type=float, dest='validation_ratio',
    #                     help='validation_ratio (default: %(default)s)', default=VALIDATION_RATIO)

    # else
    parser.add_argument('-log', '--log_dir', type=str, dest='log_dir',
                        help='log directory to save (default: %(default)s)', default='logs')
    parser.add_argument('-level', '--log_level', choices=['debug', 'info', 'warning', 'error'],
                        help='(default: %(default)s)', default='debug')

    return parser

def check_opts(opts):
    # exists(opts.data_dir, "data dir not found!")
    assert opts.n_hidden1 > 0
    assert opts.n_hidden2 > 0
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.dropout_rate >= 0
    assert opts.learning_rate >= 0
    assert opts.training_ratio > 0
    # assert opts.validation_ratio > 0

def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)
    setup_log(options)

    # args_pre = [
    #     options.data_dir,
    # ]

    kwargs_pre = {
        "train_ratio": options.training_ratio,
        # "val_ratio": options.validation_ratio,
    }

    X_train, y_train, X_val, y_val = preprocess(**kwargs_pre)

    args = [
        X_train, y_train,
        X_val, y_val,
    ]

    kwargs = {
        "n_hidden1": options.n_hidden1,
        "n_hidden2": options.n_hidden2,
        "epochs": options.epochs,
        "batch_size": options.batch_size,
        "dropout_rate": options.dropout_rate,
        "learning_rate": options.learning_rate
    }

    optimize(*args, **kwargs)

if __name__ == '__main__':
    main()