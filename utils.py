import time
import os
import logging
from logging import debug, info, warning, error

def get_model_name(options):
    return 'hidden{}-hidden{}'.format(options.n_hidden1, options.n_hidden2)

def get_param_name(options):
    return 'bs{}-dr{}-lr{}'.format(options.batch_size, options.dropout_rate, options.learning_rate)

def get_time():
    now = time.localtime()
    return "%02d_%02d_%02d_%02d_%02d_%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

def get_log_name(options):
    return get_model_name(options) + '-' + get_param_name(options) + '-' + get_time() + '.dat'

def setup_log(options):
    os.makedirs(options.log_dir, exist_ok=True)
    log_name = get_log_name(options)
    log_path = os.path.join(options.log_dir, log_name)

    logging.root.handlers = []
    log_format = '%(asctime)s %(levelname)s - %(message)s'
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    log_level = _covert_str_to_level(options)

    logging.basicConfig(format=log_format, datefmt=log_datefmt, level=log_level, filename=log_path, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(log_level)
    formatter = logging.Formatter(log_format, datefmt=log_datefmt)
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

def _covert_str_to_level(options):
    if options.log_level == 'debug':
        log_level = logging.DEBUG
    elif options.log_level == 'info':
        log_level = logging.INFO
    elif options.log_level == 'warning':
        log_level = logging.WARNING
    elif options.log_level == 'error':
        log_level = logging.ERROR
    return log_level

def exists(p, msg):
    assert os.path.exists(p), msg

def setup_log_test(options):
    os.makedirs(options.output_dir, exist_ok=True)
    log_name = 'evaluation-' + get_time() + '.dat'
    log_path = os.path.join(options.output_dir, log_name)

    logging.root.handlers = []
    log_format = '%(asctime)s %(levelname)s - %(message)s'
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    log_level = _covert_str_to_level(options)

    logging.basicConfig(format=log_format, datefmt=log_datefmt, level=log_level, filename=log_path, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(log_level)
    formatter = logging.Formatter(log_format, datefmt=log_datefmt)
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)