import numpy as np
from easydict import EasyDict as edict

__all__ = ['cfg', 'cfg_from_file']

__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN = edict()


__C.TRAIN.DATASETS = []

__C.TRAIN.NET_NAME = ''

__C.TRAIN.RESUME = ''


__C.TRAIN.SEED = 223

__C.TRAIN.START_EPOCH = 0
# Batch size per GPU
__C.TRAIN.BATCH_SIZE = 1

__C.TRAIN.INPUT_SIZE = 512

__C.TRAIN.TRAINING_SCALES = [1.0]

# Number of training epochs
__C.TRAIN.NUM_EPOCH = 32

# Do snapshot each SNAPSHOT_FREQUENCY epoch
__C.TRAIN.SNAPSHOT_FREQUENCY = 10


# Number of gpus use for training
__C.TRAIN.NUM_GPUS = 1

__C.TRAIN.USE_ALL_GPUS = False

# if == 0, disable use LR_DECAY_EPOCH
__C.TRAIN.LR_DECAY_PERIOD = 0

__C.TRAIN.LR_DECAY_EPOCH = [160, 200]


__C.TRAIN.NUM_TRAINING_SAMPLES = 1281167

__C.TRAIN.EDGE = False

__C.TRAIN.OPTIMIZER = 'SGD'

__C.TRAIN.GRADIENT_CENTRALIZATION = False

# step, poly and cosine
__C.TRAIN.LR_MODE = 'step'

__C.TRAIN.LR = 0.1

__C.TRAIN.LR_DECAY = 0.1

__C.TRAIN.WARMUP_EPOCH = 0

__C.TRAIN.WD = 0.0001

__C.TRAIN.MOMENTUM = 0.9

# Enable batch normalization or not in vgg. default is false.
__C.TRAIN.BATCH_NORM = False
# Use SE layers or not in resnext. default is false.
__C.TRAIN.USE_SE = False

# Whether to initialize the gamma of the last BN layer in each bottleneck to zero
__C.TRAIN.LAST_GAMMA = False
# dtype for training
# __C.TRAIN.DTYPE = 'float32'

__C.TRAIN.CONFIG_PATH = 'config'

__C.TRAIN.NUM_WORKERS = 8

__C.TRAIN.VAL_RESIZE = 256

__C.TRAIN.USE_PRETRAINED = False

# Mode in which to train the model. options are symbolic, imperative, hybrid
__C.TRAIN.MODE = 'hybrid'

# Number of batches to wait before logging.
__C.TRAIN.LOG_INTERVAL = 50

# Frequency of model saving.
__C.TRAIN.SNAPSHOT_FREQUENCY = 10

__C.TRAIN.VAL_FREQUENCY = 1


__C.TRAIN.EXP_PATH = ''

__C.TRAIN.VALIDATION_RESULTS_PATH = 'validation_results'

__C.TRAIN.SNAPSHOT_PATH = 'snapshots'

__C.TRAIN.LOGGING_FILE = 'logging.txt'


# Augmentations
__C.TRAIN.EXPAND_MAX_RATIO = 4.0

__C.TRAIN.MEAN_RGB = [0.485, 0.456, 0.406]

__C.TRAIN.STD_RGB = [0.229, 0.224, 0.225]

__C.TRAIN.MAX_RANDOM_AREA = 1.0

__C.TRAIN.MIN_RANDOM_AREA = 0.08

__C.TRAIN.MAX_ASPECT_RATIO = 4.0 / 3.0


__C.TRAIN.FEATURES = []

__C.TRAIN.BACKBONE_CHANNELS = []

__C.TRAIN.FILTERS = []

__C.TRAIN.STEP_MULTIPLIER = 4

__C.TRAIN.STEPS = []

__C.TRAIN.PRETRAINED_BASE = True


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            continue
            # raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            elif b[k] is None:
                pass
            else:
                raise ValueError('Type mismatch ({} vs. {}) for config key: {}'.format(type(b[k]), type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)


if __name__ == "__main__":
    pass
