import os
import json
import time
import torch

from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torchvision import ops, transforms

from utils.utils import apply_anchors, compute_pr_auc_val, gen_anc


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_loop(gpu, model, loss_func, epoch, epoch_size, optim, train_dataloader, cfg, logging):
    cls_metric_running = 0
    reg_metric_running = 0
    num_batches = epoch_size // cfg.TRAIN.BATCH_SIZE
    btic = time.time()
    for nbatch, batch in enumerate(train_dataloader):
        data = batch[0]["data"]
        target_boxes = Variable(batch[0]['bboxes'].transpose(1, 2).contiguous(), requires_grad=False)
        target_cls = Variable(batch[0]['labels'].type(torch.cuda.LongTensor), requires_grad=False)
        # target_cls = batch[0]['labels'].type(torch.cuda.LongTensor)

        output = model(data)
        pred_cls = output[:, :2, :]
        pred_boxes = output[:, 2:, :]

        cls_loss, box_loss, loss = loss_func(pred_cls, pred_boxes, target_cls, target_boxes)
        cls_metric_running += cls_loss.item()
        reg_metric_running += box_loss.item()
        loss.backward()
        optim.step()
        optim.zero_grad()

        if (nbatch + 1) % cfg.TRAIN.LOG_INTERVAL == 0 and gpu == 0:
            logging.info(
                'Epoch[{}] Batch [{}/{}]\tSpeed: {:.6f} samples/sec\t{}={:.6f}\t{}={:.6f}\tlr={:.6f}'.format(
                    epoch, nbatch, num_batches, cfg.TRAIN.BATCH_SIZE * cfg.TRAIN.LOG_INTERVAL / (time.time() - btic),
                    'CrossEntropy', cls_metric_running / cfg.TRAIN.LOG_INTERVAL, 'SmoothL1',
                                                reg_metric_running / cfg.TRAIN.LOG_INTERVAL, get_lr(optim)))
            reg_metric_running = 0
            cls_metric_running = 0
            btic = time.time()
    return


def load_checkpoint(model, checkpoint):
    """
    Load model from checkpoint.
    """
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    # remove proceeding 'N.' from checkpoint that comes from DDP wrapper
    saved_model = od["model"]
    model.load_state_dict(saved_model)


def validate(gpu, model, val_dataset_path, epoch, path_to_save, cfg):
    model.cuda(gpu)
    model.eval()

    if os.path.exists(os.path.join(path_to_save, 'aucs.json')):
        aucs = json.load(open(os.path.join(path_to_save, 'aucs.json'), 'r'))
    else:
        aucs = {}

    path_to_imgs = val_dataset_path[0]
    path_to_markup = val_dataset_path[1]
    content = json.load(open(path_to_markup))
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    result = {}

    for img_name in tqdm(content):

        img = Image.open(os.path.join(path_to_imgs, img_name)).convert('RGB')
        size = img.size

        anchors = torch.tensor(gen_anc(steps=cfg.TRAIN.STEPS, size=cfg.TRAIN.STEP_MULTIPLIER,
                                       input_size=(size[1], size[0]))).cuda(gpu)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=mean, std=std)(img)
        img = img[None, :].cuda(gpu)

        with torch.no_grad():
            output = model.module(img)[0]

        torch.cuda.synchronize(gpu)

        scores = torch.softmax(output[:2, :], dim=0)[1]
        boxes = apply_anchors(anchors, output[2:, :].T, format='corner')

        scores, indices = torch.topk(scores, k=min(100, scores.shape[0]))
        boxes = boxes[indices]
        keep = ops.nms(boxes, scores, iou_threshold=0.35)
        boxes = boxes[keep]
        scores = scores[keep]

        result[img_name] = {'objects': [], 'width': size[1], 'height': size[0]}
        for box, score in zip(boxes, scores):
            if score < 0.01:
                break
            x, y, xmax, ymax = box.cpu().detach().numpy()
            w = xmax - x
            h = ymax - y
            if w <= 0 or h <= 0 or x >= size[1] or y >= size[0]:
                continue
            if float(score) > 0.5:
                result[img_name]['objects'].append({'x': int(round(x)),
                                                    'y': int(round(y)),
                                                    'w': int(round(w)),
                                                    'h': int(round(h)),
                                                    'score': float(score)})

    name = 'epoch_{}_val.json'.format(epoch)
    json.dump(result, open(os.path.join(path_to_save, name), 'w'), indent=4)
    auc = compute_pr_auc_val(path_to_markup, os.path.join(path_to_save, name))
    os.remove(os.path.join(path_to_save, name))
    aucs['Epoch {}'.format(epoch)] = auc
    model.train()
    return auc
