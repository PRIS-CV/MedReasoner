import re
import json
import math
import numpy as np


def extract_think_bbox_points(predict_str, think="True"):
    if think == "True":
        # === 1. Extract <think>
        think_match = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL)
        think_text = think_match.group(1).strip() if think_match else None
    else:
        think_text = ""

    # === 2. Extract <answer>
    answer_match = re.search(r"<answer>\s*({.*?})\s*</answer>", predict_str, re.DOTALL)
    answer_str = answer_match.group(1).strip().replace("'", '"')
    decoder = json.JSONDecoder()
    answer_json, _ = decoder.raw_decode(answer_str)

    # === 3. Extract bbox
    pred_bbox = answer_json.get("bbox", None)
    
    # === 4. Extract points_1 and points_2
    pred_point1 = answer_json.get("points_1")
    pred_point2 = answer_json.get("points_2")
    pred_points = [pred_point1, pred_point2]
    
    return think_text, pred_bbox, pred_points


def compute_iou(pred, gt):
    x1_min, y1_min, x1_max, y1_max = pred
    x2_min, y2_min, x2_max, y2_max = gt
    inter_xmin, inter_ymin = max(x1_min, x2_min), max(y1_min, y2_min)
    inter_xmax, inter_ymax = min(x1_max, x2_max), min(y1_max, y2_max)
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    return inter_area / (area1 + area2 - inter_area + 1e-6)

def compute_pdice(pred, gt):
    def cal_circle(p1, p2):
        center_x = (p1[0] + p2[0]) / 2
        center_y = (p1[1] + p2[1]) / 2
        radius = np.linalg.norm(np.array(p1) - np.array(p2)) / 2
        return center_x, center_y, radius

    def cal_inter_area(r1, r2, d):
        if d >= r1 + r2:  # no overlap
            return 0.0
        elif d <= abs(r1 - r2):  # one inside the other
            return math.pi * min(r1, r2) ** 2
        else:
            part1 = r1**2 * math.acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
            part2 = r2**2 * math.acos((d**2 + r2**2 - r1**2) / (2 * d * r2))
            part3 = 0.5 * math.sqrt(
                (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
            )
            return part1 + part2 - part3
        
    P, G = np.array(pred), np.array(gt)
    pred_cx, pred_cy, pred_r = cal_circle(P[0], P[1])
    gt_cx, gt_cy, gt_r = cal_circle(G[0], G[1])

    d = np.linalg.norm([pred_cx - gt_cx, pred_cy - gt_cy])
    inter_area = cal_inter_area(pred_r, gt_r, d)
    union_area = math.pi * pred_r**2 + math.pi * gt_r**2 - inter_area

    dice = (2 * inter_area) / (union_area + 1e-5)
    return dice

def compute_dice(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum() + 1e-6)


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
