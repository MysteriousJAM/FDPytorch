import torch

import numpy as np

import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.plugin_manager as plugin_manager

from math import ceil
from nvidia.dali.pipeline import Pipeline


plugin_manager.load_library('.dali/libcatss.so')


def gen_dali_anc(steps, size=4, input_size=(128, 128)):
    anchors = []
    for i in range(len(steps)):
        step = steps[i]
        max_x = ceil(input_size[1] / step)
        max_y = ceil(input_size[0] / step)
        for y in range(max_y):
            for x in range(max_x):
                cx = (x + 0.5) * step
                cy = (y + 0.5) * step
                w = step * size
                anchors.append([cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2])
    return np.array(anchors)


class SSDDALIPipeline(Pipeline):
    def __init__(self, batch_size, device_id, num_threads, seed, train_dataset_path, num_gpus, data_shape, anchors,
                 cfg):
        super(SSDDALIPipeline, self).__init__(
            batch_size=batch_size,
            device_id=device_id,
            num_threads=num_threads,
            seed=seed)

        if torch.distributed.is_initialized():
            shard_id = torch.distributed.get_rank()
        else:
            shard_id = 0
        self.data_shape = data_shape

        self.input = ops.COCOReader(
            file_root=train_dataset_path[0],
            annotations_file=train_dataset_path[1],
            skip_empty=True,
            shard_id=shard_id,
            num_shards=num_gpus,
            ratio=True,
            ltrb=True,
            lazy_init=True,
            shuffle_after_epoch=True
        )

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        # Augumentation techniques

        # color distortion
        self.brightness = ops.BrightnessContrast(device='gpu', brightness=1, contrast=1)
        self.contrast = ops.BrightnessContrast(device='gpu', brightness_shift=0, brightness=1)
        self.saturation = ops.Hsv(device='gpu', hue=0)
        self.hue = ops.Hsv(device='gpu', saturation=1)

        # random expand
        self.paste = ops.Paste(device="gpu", fill_value=[m * 255 for m in cfg.TRAIN.MEAN_RGB])
        self.bbpaste = ops.BBoxPaste(device="cpu", ltrb=True)

        # random crop
        self.crop = ops.RandomBBoxCrop(
            all_boxes_above_threshold=False,
            device="cpu",
            aspect_ratio=[1 / cfg.TRAIN.MAX_ASPECT_RATIO, cfg.TRAIN.MAX_ASPECT_RATIO],
            thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
            scaling=[cfg.TRAIN.MIN_RANDOM_AREA, cfg.TRAIN.MAX_RANDOM_AREA],
            # ltrb=True,
            bbox_layout='xyXY',
            allow_no_crop=True,
            num_attempts=100
        )

        self.slice = ops.Slice(device="gpu")

        self.resize = ops.Resize(
            device="gpu",
            resize_x=data_shape,
            resize_y=data_shape,
            min_filter=types.DALIInterpType.INTERP_TRIANGULAR)

        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            crop=(data_shape, data_shape),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            # output_dtype=types.FLOAT,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            pad_output=False)
        self.bbflip = ops.BbFlip(device="cpu", ltrb=True)

        self.catss = ops.CATSS(
            device="cpu",
            anchors=list(anchors.flatten()),
            offset=True,
            stds=[0.1, 0.1, 0.2, 0.2],
            anchor_strides=cfg.TRAIN.STEPS,
            image_size=data_shape
        )

        # random variables
        self.brightness_shift = ops.Uniform(range=[-0.25, 0.25])
        self.contrast_delta = ops.Uniform(range=[0.5, 1.5])
        self.saturation_delta = ops.Uniform(range=[0.5, 1.5])
        self.hue_delta = ops.Uniform(range=[-0.5, 0.5])

        self.paste_pos = ops.Uniform(range=[0, 1])
        self.paste_ratio = ops.Uniform(range=[1, cfg.TRAIN.EXPAND_MAX_RATIO / 2])

        self.do_brightness = ops.CoinFlip(probability=0.5)
        self.do_contrast = ops.CoinFlip(probability=0.5)
        self.do_hue = ops.CoinFlip(probability=0.5)
        self.do_saturation = ops.CoinFlip(probability=0.5)
        self.do_expand = ops.CoinFlip(probability=0.5)
        self.do_flip = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        """
        Define the DALI graph.
        """
        images, bboxes, labels = self.input(name="Reader")
        images = self.decode(images)

        images = self.brightness(images, brightness_shift=self.brightness_shift() * self.do_brightness())
        images = self.contrast(images, contrast=1 + self.do_contrast() * (self.contrast_delta() - 1))
        images = self.saturation(images, saturation=1 + self.do_saturation() * (self.saturation_delta() - 1))
        images = self.hue(images, hue=self.do_hue() * self.hue_delta())

        # Paste and BBoxPaste need to use same scales and positions
        ratio = self.paste_ratio()
        px = self.paste_pos()
        py = self.paste_pos()

        do_expand = self.do_expand()
        images = self.paste(images, paste_x=px * do_expand, paste_y=py * do_expand, ratio=1 + do_expand * (ratio - 1))
        bboxes = self.bbpaste(bboxes, paste_x=px * do_expand, paste_y=py * do_expand, ratio=1 + do_expand * (ratio - 1))

        crop_begin, crop_size, bboxes, labels = self.crop(bboxes, labels)
        images = self.slice(images, crop_begin, crop_size)

        images = self.resize(images)

        do_flip = self.do_flip()
        images = self.normalize(images, mirror=do_flip)
        bboxes = self.bbflip(bboxes, horizontal=do_flip)

        bboxes, labels = self.catss(bboxes * self.data_shape, labels)

        return images, bboxes.gpu(), labels.gpu()


class DetectionDALI(object):
    """DALI partial pipeline with COCO Reader and loader. To be passed as
    a parameter of a DALI transform pipeline.

    Parameters
    ----------
    num_shards: int
         DALI pipeline arg - Number of pipelines used, indicating to the reader
         how to split/shard the dataset.
    shard_id: int
         DALI pipeline arg - Shard id of the pipeline must be in [0, num_shards).
    file_root
        Directory containing the COCO dataset.
    annotations_file
        The COCO annotation file to read from.
    device_id: int
         GPU device used for the DALI pipeline.
    """

    def __init__(self, num_shards, shard_id, file_root, annotations_file, device_id):
        self.input = ops.COCOReader(
            file_root=file_root,
            annotations_file=annotations_file,
            skip_empty=True,
            shard_id=shard_id,
            num_shards=num_shards,
            ratio=True,
            ltrb=True,
            shuffle_after_epoch=True
        )

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        # We need to build the COCOReader ops to parse the annotations
        # and have acces to the dataset size.
        class DummyMicroPipe(Pipeline):
            """ Dummy pipeline which sole purpose is to build COCOReader
            and get the epoch size. To be replaced by DALI standalone op, when available.
            """

            def __init__(self, device_id):
                super(DummyMicroPipe, self).__init__(batch_size=1,
                                                     device_id=device_id,
                                                     num_threads=1)
                self.input = ops.COCOReader(
                    file_root=file_root,
                    annotations_file=annotations_file)

            def define_graph(self):
                inputs, bboxes, labels = self.input(name="Reader")
                return inputs, bboxes, labels

        micro_pipe = DummyMicroPipe(device_id=device_id)
        micro_pipe.build()
        self._size = micro_pipe.epoch_size(name="Reader")
        del micro_pipe

    def __call__(self):
        """Returns three DALI graph nodes: inputs, bboxes, labels.
        To be called in `define_graph`.
        """
        inputs, bboxes, labels = self.input(name="Reader")
        images = self.decode(inputs)
        return images, bboxes, labels

    def size(self):
        """Returns size of COCO dataset
        """
        return self._size
