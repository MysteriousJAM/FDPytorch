import os
import time
import torch
import shutil
import logging
import datetime

import numpy as np
import os.path as osp

from argparse import ArgumentParser
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel import DistributedDataParallel as DDP

from base_config import cfg, cfg_from_file
from utils.data_loaders import get_dali_dataloader
from utils.loss import SSDLoss
from utils.ssd import get_ssd
from utils.utils import tencent_trick
from utils.training import load_checkpoint, train_loop, validate


def create_logger(exp_path):
    logger = logging.getLogger()
    handler = logging.FileHandler(osp.join(exp_path, cfg.TRAIN.LOGGING_FILE))
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


def make_exp_dirs(timestamp, override=False):
    if cfg.TRAIN.EXP_PATH == '':
        raise ValueError("You have to set experiment path.")
    exp_path = osp.join(cfg.TRAIN.EXP_PATH, timestamp)
    if osp.exists(exp_path) and override:
        shutil.rmtree(exp_path)
    elif osp.exists(exp_path) and not override:
        raise OSError('Folder {} already exists.'.format(exp_path))

    os.makedirs(exp_path)

    snapshot_path = osp.join(cfg.TRAIN.EXP_PATH, timestamp, cfg.TRAIN.SNAPSHOT_PATH)
    config_path = osp.join(osp.join(cfg.TRAIN.EXP_PATH, timestamp, cfg.TRAIN.CONFIG_PATH))
    validation_path = osp.join(osp.join(cfg.TRAIN.EXP_PATH, timestamp, cfg.TRAIN.VALIDATION_RESULTS_PATH))
    os.makedirs(snapshot_path)
    os.makedirs(config_path)
    os.makedirs(validation_path)

    return exp_path, snapshot_path, config_path, validation_path


def copy_config(config_path):
    shutil.copyfile('base_config.py', osp.join(config_path, 'base_config.py'))
    shutil.copyfile('config.yml', osp.join(config_path, 'config.yml'))


def train(gpu, train_loop_func, snapshot_path, cfg):
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    if distributed:
        torch.cuda.set_device(gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        num_gpus = torch.distributed.get_world_size()
    else:
        num_gpus = 1

    if cfg.TRAIN.SEED is None:
        cfg.TRAIN.SEED = np.random.randint(1e4)
    if distributed:
        cfg.TRAIN.SEED = (cfg.TRAIN.SEED + torch.distributed.get_rank()) % 2 ** 32
    torch.manual_seed(cfg.TRAIN.SEED)
    np.random.seed(seed=cfg.TRAIN.SEED)

    train_dataset_path = ()
    val_dataset_path = {}

    for dataset in cfg.TRAIN.DATASETS:
        if dataset.TYPE == 'TRAIN':
            train_dataset_path = (dataset.IMG_BASE_PATH, dataset.MARKUP_PATH)
        elif dataset.TYPE == 'VAL':
            val_dataset_path = (dataset.IMG_BASE_PATH, dataset.MARKUP_PATH)
        else:
            raise ValueError('Invalid dataset type: {}'.format(dataset.TYPE))
    train_loader, epoch_size = get_dali_dataloader(cfg, train_dataset_path, gpu, num_gpus,
                                                   local_seed=cfg.TRAIN.SEED - 2 ** 31)

    model = get_ssd(cfg.TRAIN.NET_NAME, trans_filters=cfg.TRAIN.FILTERS,
                    backbone_feature_filters=cfg.TRAIN.BACKBONE_CHANNELS,
                    pretrained_backbone=cfg.TRAIN.PRETRAINED_BASE)
    model = model.cuda()
    loss = SSDLoss().cuda()
    if cfg.TRAIN.OPTIMIZER.lower() == 'sgd':
        optimizer = torch.optim.SGD(tencent_trick(model), cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WD)
    elif cfg.TRAIN.OPTIMIZER.lower() == 'adam':
        optimizer = torch.optim.Adam(tencent_trick(model), cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD)
    else:
        raise NotImplementedError
    scheduler = MultiStepLR(optimizer=optimizer, milestones=cfg.TRAIN.LR_DECAY_EPOCH, gamma=cfg.TRAIN.LR_DECAY)
    if distributed:
        model = DDP(model, device_ids=[gpu], output_device=gpu)

    start_epoch = 0
    # iteration = 0
    if cfg.TRAIN.RESUME != '':
        if osp.isfile(cfg.TRAIN.RESUME):
            load_checkpoint(model.module, cfg.TRAIN.RESUME)
            checkpoint = torch.load(cfg.TRAIN.RESUME,
                                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            start_epoch = checkpoint['epoch']
            # iteration = checkpoint['iteration']
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('Provided checkpoint is not path to a file')
            return

    total_time = 0
    num_epochs = len(range(start_epoch, cfg.TRAIN.NUM_EPOCH))
    for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCH):
        start_epoch_time = time.time()
        train_loop_func(gpu, model, loss, epoch, epoch_size, optimizer, train_loader,
                                                        cfg, logging)
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time
        scheduler.step()
        if gpu == 0:
            throughput = int(epoch_size / end_epoch_time)
            logging.info(
                '[Epoch {}] speed: {} samples/sec\ttime cost: {:.6f}'.format(epoch, throughput, end_epoch_time))

        if (epoch + 1) % cfg.TRAIN.SNAPSHOT_FREQUENCY == 0 and gpu == 0:
            print("saving model...")
            obj = {'epoch': epoch + 1,
                   'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(),
                   'model': model.module.state_dict()}
            save_path = os.path.join(snapshot_path, f'epoch_{epoch}.pt')
            torch.save(obj, save_path)

        if (epoch + 1) % cfg.TRAIN.VAL_FREQUENCY == 0 and gpu == 0:
            result = validate(gpu, model, val_dataset_path, epoch, validation_path, cfg)
            logging.info('[Epoch {}] Validation result {}'.format(epoch, result))
        train_loader.reset()
    if gpu == 0:
        logging.info(
            'Average speed: {} samples/sec\tTotal time cost: {:.6f}'.format(num_epochs * epoch_size / total_time,
                                                                            total_time))


if __name__ == '__main__':
    parser = ArgumentParser(description="Train Detector")
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')
    args = parser.parse_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))

    cfg_from_file('config.yml')
    if args.local_rank == 0:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
        exp_path, snapshot_path, config_path, validation_path = make_exp_dirs(timestamp, override=True)
        copy_config(config_path)
        create_logger(exp_path)
    else:
        exp_path, snapshot_path, config_path, validation_path = [None] * 4

    if cfg.TRAIN.SEED is None:
        cfg.TRAIN.SEED = np.random.randint(1e4)

    train(args.local_rank, train_loop, snapshot_path, cfg)
