from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.logging.neptune import NeptuneLogger
from comet_ml import Experiment

import argparse
import configparser
from pprint import pprint
import time
import os

from network import Net


def parse_args():
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--layer', type=str, default='1')
    parser.add_argument('--cfg', type=str, default='./configs/default.cfg')
    args = parser.parse_args()
    assert os.path.isdir('./layer{}'.format(args.layer))
    assert os.path.isfile('./layer{0}/train.embed.layer-{0}'.format(args.layer))
    assert os.path.isfile(args.cfg)
    args._start_time = time.ctime()
    return args


def parse_config(args):
    config = configparser.ConfigParser()
    config.read(args.cfg)

    args.input_size = config['Net'].getint('input_size')
    assert args.input_size

    args.hidden_size = config['Net'].getint('hidden_size')
    assert args.hidden_size

    args.output_size = config['Net'].getint('output_size')
    assert args.output_size

    args.loss_type = config['Optim'].get('loss_type', fallback='adam')
    assert args.loss_type in ['cross-entropy']

    args.lr = config['Hyperparams'].getfloat('lr', fallback=0.01)
    args.batch_size = config['Hyperparams'].getint('batch_size', fallback=64)
    args.no_cuda = config['Hyperparams'].getboolean('no_cuda', fallback=False)
    args.seed = config['Hyperparams'].getint('seed', fallback=123)
    args.num_iterations = config['Hyperparams'].getint('num_iterations', fallback=1000)
    args.momentum = config['Hyperparams'].getfloat('momentum', fallback=0.9)
    args.multi_gpu = config['Hyperparams'].getboolean('multi_gpu', fallback=False)

    args.model_dir = config['Outputs']['model_dir']
    args.log_file = os.path.join(args.model_dir, 'train.log')
    args.results_pkl = os.path.join(args.model_dir, 'results.p')

    return args


if __name__ == "__main__":
    args = parse_args()
    args = parse_config(args)
    if os.path.exists(args.log_file):
        os.remove(args.log_file)
    pprint(vars(args))
    model = Net(args)

    wandb_logger = WandbLogger(name='layer-1',project='test')

    trainer = Trainer(gpus=1, early_stop_callback=True, logger=wandb_logger, num_sanity_val_steps=0, progress_bar_refresh_rate=0)
    trainer.fit(model)
    trainer.test()

