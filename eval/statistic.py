import json
import argparse
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from datasets import load_dataset

from utils import compute_dice, load_json, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation_model_path", type=str, help="Path to the segmentation model.")
    parser.add_argument("--data_path", type=str, help="Path to the test dataset.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to merged_output.json")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save statistics JSON")
    return parser.parse_args()

def process_model(segmentation_model_path):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'sam_zoo')))

    model_name = segmentation_model_path.split("/")[-2]
    
    if model_name == "medsam-vit-base":
        from sam_zoo.segment_anything import sam_model_registry, SamPredictor
        model_type = "vit_b"
        sam = sam_model_registry[model_type](checkpoint=segmentation_model_path).to(device='cuda')
        predictor = SamPredictor(sam)
        model = 'MedSAM'
        
    elif model_name == "SAM-Med2D_model":
        from argparse import Namespace
        from sam_zoo.sam_med2d import sam_model_registry, SammedPredictor
        args = Namespace()
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = segmentation_model_path
        sam = sam_model_registry["vit_b"](args).to('cuda')
        predictor = SammedPredictor(sam)
        model = 'SAM-Med2D'
        
    elif model_name == "MedSAM2":
        from sam_zoo.sam2.build_sam import build_sam2
        from sam_zoo.sam2.sam2_image_predictor import SAM2ImagePredictor
        from hydra.core.global_hydra import GlobalHydra
        from hydra import initialize
        GlobalHydra.instance().clear()
        config_path = '../sam_zoo/sam2'
        config_name = 'sam2_hiera_t512.yaml'
        initialize(config_path=config_path)
        predictor = SAM2ImagePredictor(build_sam2(config_name, segmentation_model_path))
        model = 'MedSAM2'

    else:
        raise ValueError(f"Unknown model name: {model}")
    
    return predictor, model

def compute_statistics(data):
    ious, pdices = [], []
    dice_points, dice_bbox, dice_joint = [], [], []
    failed_count = 0

    supercat_scores = defaultdict(lambda: {
        "ious": [],
        "pdices": [],
        "dice_points": [],
        "dice_bbox": [],
        "dice_joint": [],
        "categories": defaultdict(lambda: {
            "ious": [],
            "pdices": [],
            "dice_points": [],
            "dice_bbox": [],
            "dice_joint": []
        })
    })

    for item in data:
        is_failed = (
            item.get("think") == "error" and
            float(item.get("iou", 0.0)) == 0.0 and
            float(item.get("pdice", 0.0)) == 0.0
        )

        if is_failed:
            failed_count += 1

        try:
            iou = float(item.get("iou", 0.0))
            pdice = float(item.get("pdice", 0.0))
            d_point = float(item.get("dice_points", 0.0))
            d_bbox = float(item.get("dice_bbox", 0.0))
            d_joint = float(item.get("dice_joint", 0.0))
        except Exception as e:
            print(f"[!] Error parsing item {item.get('id', 'N/A')}: {e}")
            failed_count += 1
            iou = pdice = d_point = d_bbox = d_joint = 0.0

        ious.append(iou)
        pdices.append(pdice)
        dice_points.append(d_point)
        dice_bbox.append(d_bbox)
        dice_joint.append(d_joint)

        supercat = item.get("supercategory", "unknown")
        cat = item.get("category", "unknown")

        supercat_scores[supercat]["ious"].append(iou)
        supercat_scores[supercat]["pdices"].append(pdice)
        supercat_scores[supercat]["dice_points"].append(d_point)
        supercat_scores[supercat]["dice_bbox"].append(d_bbox)
        supercat_scores[supercat]["dice_joint"].append(d_joint)

        supercat_scores[supercat]["categories"][cat]["ious"].append(iou)
        supercat_scores[supercat]["categories"][cat]["pdices"].append(pdice)
        supercat_scores[supercat]["categories"][cat]["dice_points"].append(d_point)
        supercat_scores[supercat]["categories"][cat]["dice_bbox"].append(d_bbox)
        supercat_scores[supercat]["categories"][cat]["dice_joint"].append(d_joint)

    def compute_avg(metric_list):
        return round(sum(metric_list) / len(metric_list), 4) if metric_list else 0.0

    def reduce_category_scores(cat_dict):
        return {
            cat: {
                "mean_iou": compute_avg(metrics["ious"]),
                "mean_pdice": compute_avg(metrics["pdices"]),
                "mean_dice_points": compute_avg(metrics["dice_points"]),
                "mean_dice_bbox": compute_avg(metrics["dice_bbox"]),
                "mean_dice_joint": compute_avg(metrics["dice_joint"]),
                "count": len(metrics["ious"])
            }
            for cat, metrics in cat_dict.items()
        }

    def reduce_supercat_scores(super_dict):
        result = {}
        for supercat, scores in super_dict.items():
            sorted_categories = dict(sorted(
                reduce_category_scores(scores["categories"]).items(),
                key=lambda item: item[0].lower()
            ))

            result[supercat] = {
                "mean_iou": compute_avg(scores["ious"]),
                "mean_pdice": compute_avg(scores["pdices"]),
                "mean_dice_points": compute_avg(scores["dice_points"]),
                "mean_dice_bbox": compute_avg(scores["dice_bbox"]),
                "mean_dice_joint": compute_avg(scores["dice_joint"]),
                "count": len(scores["ious"]),
                "categories": sorted_categories
            }

        sorted_result = dict(sorted(result.items(), key=lambda item: item[0].lower()))
        return sorted_result

    stats = {
        "mean_iou": compute_avg(ious),
        "mean_pdice": compute_avg(pdices),
        "mean_dice_points": compute_avg(dice_points),
        "mean_dice_bbox": compute_avg(dice_bbox),
        "mean_dice_joint": compute_avg(dice_joint),
        "valid_samples": len(data) - failed_count,
        "failed_samples": failed_count,
        "total_samples": len(data),
        "per_super": reduce_supercat_scores(supercat_scores)
    }

    return stats

def main():
    args = parse_args()
    
    seg_model, model_name = process_model(args.segmentation_model_path)
    
    input_data = load_json(args.input_json)
    dataset = load_dataset(args.data_path)['test']
    id_dataset = {}
    for item in dataset:
        id_item = {
            "id": item["extra_info"]["id"],
            "image": item["images"][0],
            "mask": item["masks"][0]
        }
        id_dataset[id_item["id"]] = id_item

    results = []
    for item in tqdm(input_data, desc="Processing dataset"):
        id = item["id"]   
        try:
            image = id_dataset[id]["image"]
            gt_mask = np.array(id_dataset[id]["mask"]) / 255.0
            seg_model.set_image(np.array(image))

            bbox = item.get("bbox", None)
            points = item.get("points", None)
            assert bbox is not None and points is not None, "bbox and points must be provided"

            # 1. Only points
            masks_p, scores_p, _ = seg_model.predict(
                point_coords=np.array(points),
                point_labels=np.array([1] * len(points)),
                box=None,
            )
            mask_p = masks_p[np.argsort(scores_p)[::-1][0]]
            dice_p = compute_dice(mask_p, gt_mask)

            # 2. Only bbox
            masks_b, scores_b, _ = seg_model.predict(
                point_coords=None,
                point_labels=None,
                box=np.array(bbox),
            )
            mask_b = masks_b[np.argsort(scores_b)[::-1][0]]
            dice_b = compute_dice(mask_b, gt_mask)

            # 3. Both points and bbox
            masks_j, scores_j, _ = seg_model.predict(
                point_coords=np.array(points),
                point_labels=np.array([1] * len(points)),
                box=np.array(bbox),
            )
            mask_j = masks_j[np.argsort(scores_j)[::-1][0]]
            dice_j = compute_dice(mask_j, gt_mask)

        except Exception as e:
            print(f"[!] Error for ID {id}: {e}")
            dice_p, dice_b, dice_j = 0.0, 0.0, 0.0

        item["dice_points"] = round(dice_p, 4)
        item["dice_bbox"] = round(dice_b, 4)
        item["dice_joint"] = round(dice_j, 4)
        results.append(item)
    save_json(results, args.input_json)
    
    data = load_json(args.input_json)
    stats = compute_statistics(data)
    
    output_json = args.output_json.replace(".json", f"_{model_name.lower()}.json")
    save_json(stats, output_json)
    print(f"[✓] Statistics saved to: {args.output_json}")
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    main()
