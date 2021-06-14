import json
import torch

from os.path import exists, join
from PIL import Image
from torchvision import ops, transforms
from tqdm import tqdm

from base_config import cfg, cfg_from_file
from utils.training import load_checkpoint
from utils.ssd import get_ssd
from utils.utils import apply_anchors, gen_anc


TEST_BASES = {
    'NT': {
        'name': 'Test',
        'data': '../Datasets/Faces/',
        'marking': '../Datasets/Faces/test_2102.json'
    },
}
EXP_ROOT = 'exps'
EXPERIMENTS = {
    '2021-02-11--17-49-52': (80, 'CATSS_0.125')
}
SCALES = [1]
THRESHOLD = 0.01

if __name__ == '__main__':
    torch.cuda.set_device(0)
    for exp in EXPERIMENTS:
        cfg_from_file(join(EXP_ROOT, exp, 'config', 'config.yml'))
        model = get_ssd('mobilenet0.125', trans_filters=[128] * 2, backbone_feature_filters=[64, 128],
                        pretrained_backbone=True)
        epoch = EXPERIMENTS[exp][0]
        load_checkpoint(model, join(EXP_ROOT, exp, 'snapshots', 'epoch_{}.pt'.format(epoch)))
        model.cuda()
        model.eval()

        datasets_paths = []
        for key in TEST_BASES:
            datasets_paths.append([TEST_BASES[key]['name'], TEST_BASES[key]['data'], TEST_BASES[key]['marking']])

        for dataset in datasets_paths:
            dataset_name = dataset[0]
            path_to_imgs = dataset[1]
            path_to_markup = dataset[2]
            content = json.load(open(path_to_markup, 'r'))

            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

            for scale in SCALES:
                print('Processing {} on {} with scale {}.'.format(EXPERIMENTS[exp][1], dataset_name, scale))
                path_to_save = join('results', '_'.join((EXPERIMENTS[exp][1], str(epoch), dataset_name, 'scale',
                                                         str(scale))) + '.json')
                if exists(path_to_save):
                    saved = json.load(open(path_to_save))
                else:
                    saved = {}
                result = saved
                bad = []
                for num_imgs, img_name in enumerate(tqdm(content)):
                    if img_name in result:
                        continue
                    try:
                        img = Image.open(join(path_to_imgs, img_name)).convert('RGB')
                    except:
                        continue
                    orig_size = img.size
                    if scale != 1:
                        img = img.resize((orig_size[0] // scale, orig_size[1] // scale), resample=Image.BILINEAR)

                    dst_shape = img.size
                    if min(dst_shape) > 3000:
                        continue
                    with torch.no_grad():
                        img = transforms.ToTensor()(img)
                        img = transforms.Normalize(mean=mean, std=std)(img)
                        img_np = img.cpu().detach().numpy()
                        img = img[None, :].cuda()

                        output = model(img)[0]
                        torch.cuda.synchronize()

                        scores = torch.softmax(output[:2, :], dim=0)[1]
                        anchors = torch.tensor(gen_anc(steps=cfg.TRAIN.STEPS, size=cfg.TRAIN.STEP_MULTIPLIER,
                                                       input_size=(dst_shape[1], dst_shape[0]))).cuda()
                        boxes = apply_anchors(anchors, output[2:, :].T, format='corner')
                        scores, indices = torch.topk(scores, k=min(400, scores.shape[0]))
                        boxes = boxes[indices]
                        keep = ops.nms(boxes, scores, iou_threshold=0.35)
                        boxes = boxes[keep]
                        scores = scores[keep]

                    result[img_name] = {'objects': [], 'width': dst_shape[1], 'height': dst_shape[0]}
                    for box, score in zip(boxes, scores):
                        if score <= THRESHOLD:
                            break
                        x, y, xmax, ymax = box.cpu().detach().numpy()
                        x, y, xmax, ymax = [int(k * scale) for k in (x, y, xmax, ymax)]
                        w = xmax - x
                        h = ymax - y
                        if w <= 0 or h <= 0 or x >= orig_size[0] or y >= orig_size[1]:
                            continue
                        result[img_name]['objects'].append({'x': int(round(x)),
                                                            'y': int(round(y)),
                                                            'w': int(round(w)),
                                                            'h': int(round(h)),
                                                            'score': float(score)})
                    if (num_imgs + 1) % 5000 == 0:
                        torch.cuda.synchronize()
                        json.dump(result, open(path_to_save, 'w'), indent=4)
                        json.dump(bad, open('{}_bad.json'.format(dataset_name), 'w'), indent=4)

                torch.cuda.synchronize()
                json.dump(result, open(path_to_save, 'w'), indent=4)
                json.dump(bad, open('{}_bad.json'.format(dataset_name), 'w'), indent=4)
