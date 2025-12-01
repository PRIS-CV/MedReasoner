import re
import json
import math


def reason_format_reward(predict_think_str: str, predict_answer_str: str) -> float:
    """
    Compute a total format reward based on:
        1. <think>...</think> reasoning presence and structure
        2. <answer>...</answer> JSON structure and field correctness
    Returns:
        think_score: a float in [0.0, 1.0]
        seg_score: a float in [0.0, 1.0]
    """
    
    def think_format_reward(predict_str: str) -> float:
        if not predict_str:
            return 0.0  # No <think> block
        
        think_text = predict_str.group(1).strip()
        if not think_text:
            return 0.2  # Empty <think> block
        return 1.0  # Valid <think> block with content

    def seg_format_reward(predict_str: str) -> float:
        try:
            if not predict_str:
                return 0.0  # No <answer> block
            
            answer_text = predict_str.group(1).strip().replace("'", '"')
            answer_json = json.loads(answer_text)
        except Exception:
            return 0.2  # Invalid JSON

        if not isinstance(answer_json, dict):
            return 0.2

        required_keys = ["bbox", "points_1", "points_2"]
        if not all(k in answer_json for k in required_keys):
            return 0.4  # Missing keys

        bbox_valid, points_valid = True, True

        bbox = answer_json["bbox"]
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
            bbox_valid = False

        for key in ["points_1", "points_2"]:
            pts = answer_json[key]
            if not (isinstance(pts, list) and len(pts) == 2 and all(isinstance(v, (int, float)) for v in pts)):
                points_valid = False
                break

        if not bbox_valid and not points_valid:
            return 0.4
        elif not bbox_valid or not points_valid:
            return 0.6
        else:
            return 1.0  # Fully correct
        
    # Compute individual scores
    think_score = think_format_reward(predict_think_str)
    seg_score = seg_format_reward(predict_answer_str)

    return think_score, seg_score


def is_valid_box(box):
    return isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)

def is_valid_point(point):
    return isinstance(point, list) and len(point) == 2 and all(isinstance(v, (int, float)) for v in point)

def norm_val(val, bbox=None, max_val=2.0):
    if bbox is None:
        return min(val, max_val)
    
    else:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        diag = math.sqrt(w**2 + h**2)
        norm = val / (diag + 1e-5)
        return max(min(norm, max_val), 0.0)

def smooth_log_val(val, k=3.0):
    s_val = math.log(k * val + 1) / math.log(k + 1)

    return s_val * 0.9 + 0.1

def smooth_exp_val(val, k=3.0, c=1.0):
    raw_val = 1.0 / (1 + math.exp(k * (val - c)))
    min_val = 1.0 / (1 + math.exp(k * (2 - c)))
    max_val = 1.0 / (1 + math.exp(k * (0 - c)))
    s_val = (raw_val - min_val) / (max_val - min_val + 1e-8)
    
    return s_val * 0.9 + 0.1

def set_position_valid(p1, p2, bbox):
    p1_valid = 1.0 if bbox[0] <= p1[0] <= bbox[2] and bbox[1] <= p1[1] <= bbox[3] else 0.5
    p2_valid = 1.0 if bbox[0] <= p2[0] <= bbox[2] and bbox[1] <= p2[1] <= bbox[3] else 0.5
    
    return p1_valid * p2_valid

def set_area_valid(box1, box2):
    def area(box):
        return max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))

    area1 = area(box1)
    area2 = area(box2)

    return 1.0 if (0.5 * area2) < area1 < (2 * area2) else 0.5

def set_spread_valid(p1_valid, p2_valid):
    dist = math.dist(p1_valid, p2_valid)

    return 1.0 if dist > 10 else 0.5

def penalize_reward(reward, valid1, valid2, k=0.7):    
    avg_valid = (valid1 + valid2) / 2
    reward = k * reward + (1 - k) * reward * avg_valid

    return round(min(max(reward, 0.0), 1.0), 4)


def seg_bbox_reward(predict_str, gt, reward_mode) -> float:
    """
    Evaluate the quality of a **predicted axis-aligned bounding box** against 
    the ground-truth (GT) box.  Three complementary rewards are available:

        1) IoU-reward   – classic overlap ratio, smoothed to (0.1 , 1]  
        2) Align-reward – mean edge-wise L1 distance, normalised by GT diagonal  
        3) Scale-reward – shape consistency: √((Δlog-area)² + (Δlog-aspect)²)

    Parameters
    ----------
    predict_str : str | re.Match
        Model output that contains a JSON string with a "bbox" field.
        - If it is a regex Match object, .group(1) will be extracted.
        - Otherwise it is treated as a plain string.
    gt : dict
        Ground-truth dict with keys:
            "bbox_2d"  : list[4] – [x1, y1, x2, y2]
            "point_2d" : list[[x, y], [x, y]] – (optional) two GT points.
    reward_mode : {"soft", "hard"}
        "soft"  : only IoU reward is returned;
        "hard"  : IoU + alignment + scale rewards are returned.

    Returns
    -------
    tuple(float, float, float, float, float)
        (IoU-reward, Align-reward, Scale-reward, position_valid, area_valid)
        – In “soft” mode the two middle values are 0.0.
    """

    def iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
        area2 = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))
        union_area = area1 + area2 - inter_area

        return inter_area / (union_area + 1e-5)

    def seg_bbox_iou_reward(pred_bbox, gt_bbox):
        iou_reward = iou(pred_bbox, gt_bbox)
        reward = smooth_log_val(iou_reward, k=3.0)

        return reward

    def l1_dist(box1, box2):
        l1_score = sum(abs(a - b) for a, b in zip(box1, box2)) / 4
        
        return l1_score

    def seg_bbox_align_reward(pred_bbox, gt_bbox):
        align_score = l1_dist(pred_bbox, gt_bbox)
        align_reward = norm_val(align_score, bbox=gt_bbox, max_val=2.0)
        reward = smooth_exp_val(align_reward, k=3.0, c=1.0)

        return reward

    def scale_ratio(box1, box2):
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        d_area = math.log((w1 * h1 + 1e-6) / (w2 * h2 + 1e-6))
        d_aspt = math.log((w1 / h1 + 1e-6) / (w2 / h2 + 1e-6))

        d_s = math.sqrt(d_area ** 2 + d_aspt ** 2)
        return d_s

    def seg_bbox_scale_reward(pred_bbox, gt_bbox):
        scale_score = scale_ratio(pred_bbox, gt_bbox)
        scale_reward = norm_val(scale_score, max_val=1.0)
        reward = smooth_exp_val(scale_reward, k=3.0, c=0.5)

        return reward

    try:
        gt_bbox = gt["bbox_2d"]
        gt_points = gt.get("point_2d", [])

        # Extract <answer> block
        if not predict_str:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        answer_str = predict_str.group(1).strip().replace("'", '"')
        answer_json = json.loads(answer_str)

        pred_bbox = answer_json.get("bbox", None)
        if not is_valid_box(pred_bbox):
            return 0.0, 0.0, 0.0, 0.0, 0.0 # Invalid format
        
        # Bboxes valid check
        position_valid = set_position_valid(gt_points[0], gt_points[1], pred_bbox)
        area_valid = set_area_valid(pred_bbox, gt_bbox)
        
        # Compute IoU Reward
        iou_reward = seg_bbox_iou_reward(pred_bbox, gt_bbox)
        final_iou_reward = penalize_reward(iou_reward, position_valid, area_valid)
        
        if reward_mode == "soft":
            return final_iou_reward, 0.0, 0.0, position_valid, area_valid
        
        elif reward_mode == "hard":
            # Compute Align Reward
            align_reward = seg_bbox_align_reward(pred_bbox, gt_bbox)
            final_align_reward = penalize_reward(align_reward, position_valid, area_valid)
            
            # Compute Scale Reward
            scale_reward = seg_bbox_scale_reward(pred_bbox, gt_bbox)
            final_scale_reward = penalize_reward(scale_reward, position_valid, area_valid)

            return final_iou_reward, final_align_reward, final_scale_reward, position_valid, area_valid
        else:
            raise ValueError(f"Invalid reward mode: {reward_mode}. Expected 'hard' or 'soft'.")

    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0


def seg_points_reward(predict_str, gt, reward_mode) -> float:
    """
    Evaluate the consistency between a **pair of predicted points** (two-point
    annotation) and the ground-truth (GT) pair.  
    Three complementary rewards are provided:
        1) Dice-reward   – disc overlap of the two diameter-defined circles  
        2) Align-reward  – L1 alignment of the two endpoints (order-free)  
        3) Angle-reward  – direction consistency (0° and 180° → best, 90° → worst)

    Parameters
    ----------
    predict_str : str | re.Match
        Model output that contains a JSON string with keys
        ``"points_1": [x, y]`` and ``"points_2": [x, y]``.
        If a regex *Match* object is passed, ``group(1)`` is extracted.
    gt : dict
        Ground-truth dictionary with
            • "bbox_2d"  : [x1, y1, x2, y2] – the GT bounding box  
            • "point_2d" : [[x, y], [x, y]] – the GT point pair
    reward_mode : {"soft", "hard"}
        "soft"  – only Dice-reward is returned  
        "hard"  – Dice + Align + Angle are returned

    Returns
    -------
    tuple(float, float, float, float, float)
        (dice_reward, align_reward, angle_reward,
         position_valid_flag, spread_valid_flag)

        In *soft* mode the two middle rewards are 0.0.
    """
    def cal_circle(p1, p2):
        center_x = (p1[0] + p2[0]) / 2
        center_y = (p1[1] + p2[1]) / 2
        radius = math.dist(p1, p2) / 2 + 1e-5
        
        return center_x, center_y, radius

    def cal_inter_area(r1, r2, d):
        if d >= r1 + r2 - 1e-5:
            return 0.0

        if d <= abs(r1 - r2) + 1e-5:
            return math.pi * min(r1, r2) ** 2

        x1 = (d*d + r1*r1 - r2*r2) / (2*d*r1)
        x2 = (d*d + r2*r2 - r1*r1) / (2*d*r2)
        x1 = max(-1.0, min(1.0, x1))
        x2 = max(-1.0, min(1.0, x2))

        part1 = r1*r1 * math.acos(x1)
        part2 = r2*r2 * math.acos(x2)
        part3 = 0.5 * math.sqrt(
            max(0.0, (-d + r1 + r2)*(d + r1 - r2)*(d - r1 + r2)*(d + r1 + r2))
        )
        return part1 + part2 - part3

    def cal_union_area(r1, r2, d):
        inter_area = cal_inter_area(r1, r2, d)
        return math.pi * r1**2 + math.pi * r2**2 - inter_area

    def seg_point_dice_reward(pred_pts, gt_pts):
        pred_cx, pred_cy, pred_r = cal_circle(pred_pts[0], pred_pts[1])
        gt_cx, gt_cy, gt_r = cal_circle(gt_pts[0], gt_pts[1])

        d = math.dist((pred_cx, pred_cy), (gt_cx, gt_cy))
    
        inter_area = cal_inter_area(pred_r, gt_r, d)
        union_area = cal_union_area(pred_r, gt_r, d)
        
        dice_reward = (2 * inter_area) / (union_area + 1e-5)
        reward = smooth_log_val(dice_reward, k=3.0)
        
        return reward    

    def l1_dist(p1, p2):
        return (abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])) / 2

    def seg_point_align_reward(pred_pts, gt_pts, gt_bbox):
        d1 = l1_dist(pred_pts[0], gt_pts[0]) + l1_dist(pred_pts[1], gt_pts[1])
        d2 = l1_dist(pred_pts[0], gt_pts[1]) + l1_dist(pred_pts[1], gt_pts[0])
        align_score = min(d1, d2) / 2
        align_reward = norm_val(align_score, bbox=gt_bbox, max_val=2.0)
        reward = smooth_exp_val(align_reward)
        
        return reward

    def point_vector(p1, p2):
        return (p2[0] - p1[0], p2[1] - p1[1])

    def seg_point_angle_reward(pred_pts, gt_pts):
        vx_g, vy_g = point_vector(gt_pts[0], gt_pts[1])
        vx_p, vy_p = point_vector(pred_pts[0], pred_pts[1])
        len_g = math.hypot(vx_g, vy_g) + 1e-8
        len_p = math.hypot(vx_p, vy_p) + 1e-8
        
        cos_sim = abs((vx_g * vx_p + vy_g * vy_p) / (len_g * len_p))
        reward = smooth_log_val(cos_sim, k=3.0)

        return reward

    try:
        gt_bbox = gt["bbox_2d"]
        gt_points = gt.get("point_2d", [])
        
        # Extract <answer> block
        if not predict_str:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        answer_str = predict_str.group(1).strip().replace("'", '"')
        answer_json = json.loads(answer_str)
        
        pred_point1 = answer_json.get("points_1")
        pred_point2 = answer_json.get("points_2")
        if not is_valid_point(pred_point1) or not is_valid_point(pred_point2):
            return 0.0, 0.0, 0.0, 0.0, 0.0 # Invalid format

        # Points valid check
        position_valid = set_position_valid(pred_point1, pred_point2, gt_bbox)
        spread_valid = set_spread_valid(pred_point1, pred_point2)

        # Compute Dice reward
        dice_reward = seg_point_dice_reward([pred_point1, pred_point2], gt_points)
        final_dice_reward = penalize_reward(dice_reward, position_valid, spread_valid)

        if reward_mode == "soft":
            return final_dice_reward, 0.0, 0.0, position_valid, spread_valid

        elif reward_mode == "hard":
            align_reward = seg_point_align_reward([pred_point1, pred_point2], gt_points, gt_bbox)
            final_align_reward = penalize_reward(align_reward, position_valid, spread_valid)

            angle_reward = seg_point_angle_reward([pred_point1, pred_point2], gt_points)
            final_angle_reward = penalize_reward(angle_reward, position_valid, spread_valid)

            return final_dice_reward, final_align_reward, final_angle_reward, position_valid, spread_valid

        else:
            raise ValueError(f"Invalid reward mode: {reward_mode}. Expected 'hard' or 'soft'.")

    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0


def compute_score(predict_str: str, ground_truth: str, reward_mode: str):
    """
    Combined reward for segmentation task based on format + geometry + structure.
    """
    
    predict_think_str = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL)
    predict_answer_str = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    gt = json.loads(ground_truth)
    
    format_think_score, format_seg_score = reason_format_reward(predict_think_str, predict_answer_str)
    bbox_iou_score, bbox_align_score, bbox_scale_score, bbox_position_valid, bbox_area_valid = seg_bbox_reward(predict_answer_str, gt, reward_mode)
    points_dice_score, points_align_score, points_angle_score, points_position_valid, points_spread_valid = seg_points_reward(predict_answer_str, gt, reward_mode)
    score = format_think_score + format_seg_score \
            + bbox_iou_score + bbox_align_score + bbox_scale_score \
            + points_align_score + points_angle_score + points_dice_score

    rewards = {
        "score": score,
        # format score
        "format_think": format_think_score,
        "format_seg": format_seg_score,
        # soft score
        "bbox_iou": bbox_iou_score,
        "points_dice": points_dice_score,
        # hard score
        "bbox_align": bbox_align_score,
        "bbox_scale": bbox_scale_score,
        "points_align": points_align_score,
        "points_angle": points_angle_score,
        # valid check
        "bbox_position_valid": bbox_position_valid,
        "bbox_area_valid": bbox_area_valid,
        "points_position_valid": points_position_valid,
        "points_spread_valid": points_spread_valid,
    }
    
    return rewards
