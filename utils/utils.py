import json
import torch


from math import ceil
from bisect import bisect_right
from operator import itemgetter


def gen_anc(steps, size=4, input_size=(128, 128)):
    anchors = []

    for step in steps:
        max_x = ceil(input_size[1]/step)
        max_y = ceil(input_size[0]/step)
        for y in range(max_y):
            for x in range(max_x):
                cx = (x + 0.5) * step
                cy = (y + 0.5) * step
                anchors.append([cx, cy, step * size, step * size])
    return anchors


def apply_anchors(anchors, boxes, mean=(0, 0, 0, 0), std=(0.1, 0.1, 0.2, 0.2), format='center'):
    cx = ((boxes[:, 0] * std[0] + mean[0]) * anchors[:, 2] + anchors[:, 0])[:, None]
    cy = ((boxes[:, 1] * std[1] + mean[1]) * anchors[:, 3] + anchors[:, 1])[:, None]
    w = (torch.exp(boxes[:, 2] * std[2] + mean[2]) * anchors[:, 2])[:, None]
    h = (torch.exp(boxes[:, 3] * std[3] + mean[3]) * anchors[:, 3])[:, None]
    if format == 'center':
        return torch.cat([cx, cy, w, h], dim=-1)
    elif format == 'corner':
        return torch.cat([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    else:
        print(format)
        raise NotImplementedError


def compute_pr_auc_val(path_to_gt_json, path_to_res_json):
    """
    Helper function to read jsons and calculate auc
    :param path_to_gt_json: str
            Path to json file with ground truth bboxes
    :param path_to_res_json: str
            Path to json file with results.
    :return: float
            Area under precision-recall curve.
    """
    result = json.load(open(path_to_res_json))
    gt = json.load(open(path_to_gt_json))

    pr = compute_pr_simple(gt, result)
    auc = pr_auc(pr)

    return auc


def _overlap(side1, side2):
    """
    Helper function for calculating IoU. Sides should be parallel.
    :param side1: first side in format [start coordinate, length], for example [x, w] or [y, h]
    :param side2: second side in format [start coordinate, length], for example [x, w] or [y, h]
    :return: length of projection of side1 on side2
    """
    if side1[0] > side2[0]: # start coordinate of side1 should be less or equal to start coordinate of side2
        side1, side2 = side2, side1
    end_side1, end_side2 = map(sum, (side1, side2)) # calculation of end coordinate for sides
    if side2[0] > end_side1: # if end coordinate of side1 is less than start coordinate of side2 then the length of
        # the projection is 0
        return 0
    if end_side1 > end_side2: # if end coordinate of side1 is greater than start coordinate of side2 then the length
        # of the projection is equal to the length of side2
        return side2[1]
    return end_side1 - side2[0] # in remaining case the length of the projection is equal to the difference between
    # start coordinate of side2 and end coordinate of side1


def _iou(a, b):
    """
    Computes intersection over union characteristics between two rectangles a and b
    :param a: dictionary representing rectangle with keys ['x', 'y', 'w', 'h']
    :param b: dictionary representing rectangle with keys ['x', 'y', 'w', 'h']
    :return: area of intersection of a and b over area of union of a nad b
    """
    overlap_x = _overlap((a['x'], a['w']), (b['x'], b['w']))
    overlap_y = _overlap((a['y'], a['h']), (b['y'], b['h']))
    intersection_area = 1. * overlap_x * overlap_y
    union_area = a['w'] * a['h'] + b['w'] * b['h'] - intersection_area
    if union_area == 0:
        return
    return 1. * intersection_area / union_area


def compute_pr_simple(gt, result, iou_thresh=0.3, iou_ignore_thresh=0.3, num_points=1000):
    """
    Computes precision and recall for result
    :param gt: dictionary of ground-truth face bounding boxes with optional ignore flag.
           format: {image name : {objects: [{'x' : ,'y' : , 'w' : ,'h' : ,'ignore' : }, ...]}, ...}
           or  {image name : [{'x' : ,'y' : ,'w' :, 'h' : }], ...}
    :param result: dictionary of result face bounding boxes with confidence scores.
           format: {image name : {objects:[{'x' : , 'y' : , 'w' : ,'h' : , 'score' : }, ...]}, ...}
           or  {image name : [{'x' : , 'y' : ,'h' : , 'w' : , 'score' : }], ...}
    :param iou_thresh: threshold for IoU. If IoU between result face and ground-truth face is greater than iou_thresh
           then result face is true positive
    :param iou_ignore_thresh: threshold for IoU. If there is ignored ground-truth face such that IoU between false
           positive result face and this ignored ground-truth face is greater than iou_ignore_thresh then result face is
           excluded from false positives
    :param num_points: number of points in which precision and recall will be calculated
    :return:
    """
    tps = []  # scores of true positive result faces
    fps = []  # scores of false positive result faces
    gt_count = 0  # number of ground-truth faces
    for image_name in gt:
        # extracting ground-truth faces and ignored ground-truth faces for image_name
        gtf = gt[image_name]['objects'] if 'objects' in gt[image_name] else gt[image_name]
        gt_faces = [face for face in gtf if 'ignore' not in face or not face['ignore']]
        gt_faces_ignore = [face for face in gtf if 'ignore' in face and face['ignore']]
        gt_count += len(gt_faces)

        # extracting result faces for image_name
        res_faces = []
        if image_name in result:
            res_faces = result[image_name]['objects'] if 'objects' in result[image_name] else result[image_name]
            res_faces = sorted(res_faces, key=itemgetter('score'), reverse=True)

        fp_faces = [] # false positive faces
        # check with not ignored faces
        for res_face in res_faces:
            if len(gt_faces) <= 0:
                fp_faces.append(res_face)
                continue

            # determining for every res_face if it is true all false positive
            ious = [_iou(res_face, gt_face) for gt_face in gt_faces]
            max_index, max_iou = max(enumerate(ious), key=itemgetter(1))
            if max_iou >= iou_thresh:
                tps.append(res_face['score'])
                del gt_faces[max_index]
            else:
                fps.append(res_face['score'])

        # check with ignored faces
        for fp_face in fp_faces:
            if len(gt_faces_ignore) <= 0:
                fps.append(fp_face['score'])
            else:
                ious = [_iou(fp_face, gt_face) for gt_face in gt_faces_ignore]
                max_index, max_iou = max(enumerate(ious), key=itemgetter(1))
                if max_iou < iou_ignore_thresh:
                    fps.append(fp_face['score'])

    # list of both scores true positive and false positive
    tpfps = fps
    tpfps.extend(tps)
    tpfps = sorted(tpfps)
    tpfps_len = len(tpfps)
    tps = sorted(tps)
    tps_len = len(tps)
    step = max(1, int(tps_len / num_points))
    scores = [0.0]
    scores.extend([tps[i] for i in range(0, tps_len, step)])
    ans = []
    if gt_count == 0:
        if tps_len == 0:
            return [1]
        else:
            return [0]
    for score in scores:
        # number of false and true positives with score less then current score
        tpfp_count = tpfps_len - bisect_right(tpfps, score) + 1
        # number of true positives with score less then current score
        tp_count = tps_len - bisect_right(tps, score) + 1
        recall = float(tp_count) / gt_count
        precision = float(tp_count) / tpfp_count
        ans.append((recall, precision, score))

    return ans


def pr_auc(pr):
    auc = 0
    pr = sorted(pr, key=itemgetter(0))
    for i in range(len(pr) - 1):
        auc += (pr[i + 1][0] - pr[i][0]) * (pr[i][1] + pr[i + 1][1]) / 2
    return auc


def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}]
