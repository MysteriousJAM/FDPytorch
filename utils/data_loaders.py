from nvidia.dali.plugin.pytorch import DALIGenericIterator

from .dali import SSDDALIPipeline, gen_dali_anc


def get_dali_dataloader(cfg, train_dataset_path, device, num_gpus, local_seed):
    anchors = gen_dali_anc(cfg.TRAIN.STEPS, cfg.TRAIN.STEP_MULTIPLIER, input_size=(cfg.TRAIN.INPUT_SIZE,
                                                                                   cfg.TRAIN.INPUT_SIZE))
    pipeline = SSDDALIPipeline(batch_size=cfg.TRAIN.BATCH_SIZE, device_id=device, num_threads=cfg.TRAIN.NUM_WORKERS,
                               seed=local_seed, train_dataset_path=train_dataset_path,
                               num_gpus=num_gpus,  data_shape=cfg.TRAIN.INPUT_SIZE,
                               anchors=anchors, cfg=cfg)

    pipeline.build()
    epoch_size = pipeline.epoch_size("Reader")
    train_loader = DALIGenericIterator([pipeline], ['data', 'bboxes', 'labels'],
                                       epoch_size / num_gpus, auto_reset=False)
    return train_loader, epoch_size // num_gpus
