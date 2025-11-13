import os
import json
import argparse
from typing import List, Dict

import cv2
import numpy as np


def load_json_by_id(json_path: str) -> Dict[str, Dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def draw_bbox(image, bbox, color, label, thickness=8):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=thickness)
    # cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)


def draw_points(image, points: List[List[int]], color, label_prefix, radius=12):
    for idx, (x, y) in enumerate(points):
        cv2.circle(image, (x, y), radius=radius, color=color, thickness=-1)
        # cv2.putText(image, f"{label_prefix}_P{idx+1}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def visualize_prompt(
    gt_json_path: str,
    pred_json_path: str,
    id_list: List[str],
    image_root: str,
    output_dir: str = None,
):
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    gt_data = load_json_by_id(gt_json_path)
    pred_data = load_json_by_id(pred_json_path)

    for img_id in id_list:
        if img_id not in gt_data or img_id not in pred_data:
            print(f"[!] ID {img_id} missing in one of the files.")
            continue

        img_name = gt_data[img_id]["image"]
        img_path = os.path.join(image_root, img_name)
        if not os.path.exists(img_path):
            print(f"[!] Image not found: {img_path}")
            continue

        # Load grayscale image and convert to RGB
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image] * 3, axis=-1)
        image_gt = image.copy()

        # --- Draw GT ---
        gt_bbox = gt_data[img_id].get("bbox") or gt_data[img_id].get("bbox_2d")
        if gt_bbox:
            draw_bbox(image, gt_bbox, color=(0, 255, 0), label="GT")
            draw_bbox(image_gt, gt_bbox, color=(0, 255, 0), label="GT")
        gt_points = gt_data[img_id].get("point") or gt_data[img_id].get("point_2d")
        if gt_points:
            draw_points(image, gt_points, color=(0, 255, 0), label_prefix="GT")
            draw_points(image_gt, gt_points, color=(0, 255, 0), label_prefix="GT")

        # --- Draw Pred ---
        pred_bbox = pred_data[img_id].get("bbox")
        if pred_bbox:
            draw_bbox(image, pred_bbox, color=(255, 0, 0), label="Pred")
        pred_points = pred_data[img_id].get("point")
        if pred_points:
            draw_points(image, pred_points, color=(255, 0, 0), label_prefix="Pred")

        if output_dir:
            save_path = os.path.join(output_dir, img_id)
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            save_path_gt = os.path.join(output_dir, f"{img_id.split('.')[0]}_gt.png")
            cv2.imwrite(save_path_gt, cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_list", type=str, required=True, help="Comma-separated list of IDs or path to a txt file")
    parser.add_argument("--gt_json", type=str, required=True, help="Path to ground-truth JSON")
    parser.add_argument("--pred_json", type=str, required=True, help="Path to prediction JSON")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory of images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save visualizations")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isfile(args.id_list):
        with open(args.id_list, "r") as f:
            id_list = [line.strip() for line in f if line.strip()]
    else:
        id_list = args.id_list.split(",")

    visualize_prompt(
        gt_json_path=args.gt_json,
        pred_json_path=args.pred_json,
        id_list=id_list,
        image_root=args.image_root,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
