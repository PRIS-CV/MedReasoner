import os
import json
import argparse

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from qwen_vl_utils import process_vision_info
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils import extract_think_bbox_points, compute_iou, compute_pdice, compute_dice, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", help="Path to the reasoning model.")
    parser.add_argument("--segmentation_model_path", help="Path to the segmentation model.")
    parser.add_argument("--think", type=str, help="Whether to include thinking in the output (True/False).", default="True")
    parser.add_argument("--image_path", help="Path to the input image.")
    parser.add_argument("--mask_path", help="Path to the input mask.")
    parser.add_argument("--question", help="Question to ask the model.")
    parser.add_argument("--solution", help="Expected solution for the task.")
    parser.add_argument("--output_path", help="Path to the output file.")
    return parser.parse_args()

def load_model(model_path):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")
    
    return model, processor

def prepare_message(image, question, think="True"):
    if think == "True":
        user_prompt = (
            "<image>\n"
            "This is a medical image localization task. Modalities include X-ray, CT, MRI, ultrasound, endoscopy, fundus, pathology, dermoscopy, and mammography.\n"
            "The question provides only implicit cues about the target region. Begin by inferring its likely focus as a clinician would.\n"
            "Your goal is to use rigorous visual reasoning to identify the anatomical or pathological region implied by the question and precisely locate it in the image.\n\n"

            "Think step by step to answer the question and accurately ground the target in the image.\n"
            "Question: {Question}\n\n"
                      
            "Step-by-step guidelines:\n"
            "1. Interpreting the vague question: The question may not explicitly describe the target. Start by hypothesizing its implied intent using prior clinical knowledge and general context.\n"
            "2. Gathering visual evidence: Systematically inspect the image and extract relevant visual features—such as shape, edge definition, brightness or density, symmetry, texture, and structural heterogeneity—to validate or revise your initial hypothesis.\n"
            "3. Inferring the most likely target region: Integrate your clinical hypothesis with visual observations. Narrow down to a single region that best matches the implied intent of the question, guided by the most salient visual cues.\n"
            "4. Delivering precise localization: Express your conclusion by providing only spatial location details (bounding box and points). Omit diagnosis or classification.\n"
            "5. Resolving inconsistencies: If earlier reasoning conflicts with visible evidence, revise your interpretation and prioritize the observed visual data.\n\n"
            
            "Output Formats:\n"
            "Your response must adhere to a strict format, containing exactly one <think> block followed immediately by one <answer> block:\n"
            "- <think>...</think>: Use clinical reasoning to precisely explain how the observed visual features and relevant medical context were integrated to determine the target region in the image.\n"
            "- <answer>...</answer>: This section must contain a JSON object with the following keys and values:\n"
            "    \"bbox\": the tightest bounding box enclosing the target region.\n"
            "    \"points_1\": a primary key point within the bbox, on the target region.\n"
            "    \"points_2\": a second, distinct key point within the target region.\n\n"
            
            "Response Rules:\n"
            "- The entire output must be a single continuous string, containing precisely one <think> block and one <answer> block, with no additional text or formatting.\n"
            "- The <think> section must exclusively reflect confident clinical reasoning focused on precise localization, without any hedging, ambiguity, or expressions of uncertainty.\n"
            "- The <answer> block must not be empty. You must always output a valid bounding box and two key points.\n"
            "- The bounding box must be the tightest possible rectangle that completely encloses the target region, excluding any background pixels.\n"
            "- The two key points must have distinct coordinates, lie strictly within the target region, and correspond to different salient visual cues.\n\n"

            "Example Output:\n"
            "<think> thinking process here </think>"
            "<answer>{Answer}</answer>"
        )
    
    else:
        user_prompt = (
            "<image>\n"
            "Please answer {Question} with bbox and points.\n\n"
            
            "Output Formats:\n"
            "Your response must adhere to a strict format, containing exactly one <answer> block:\n"
            "- <answer>...</answer>: This section must contain a JSON object with the following keys and values:\n"
            "    \"bbox\": the tightest bounding box enclosing the target region.\n"
            "    \"points_1\": a primary key point within the bbox, on the target region.\n"
            "    \"points_2\": a second, distinct key point within the target region.\n\n"
            
            "Example Output:\n"
            "<answer>{Answer}</answer>"
        )
    
    return [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt.format
                (
                    Question=question,
                    Answer="{'bbox': [xmin, ymin, xmax, ymax], 'points_1': [x1, y1], 'points_2': [x2, y2]}"
                )
            }
        ]}
    ]

def process_message(model, processor, messages, device):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=text, images=img_inputs, videos=vid_inputs,
                    return_tensors="pt").to(device)

    with torch.inference_mode():
        gen_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    answer_txt = processor.decode(gen_ids[0][inputs.input_ids.shape[-1]:],
                                skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return answer_txt

def vis_mask(image, mask, color, alpha=0.5):
    overlay = image.copy()

    mask_bool = mask.astype(bool)
    mask_3ch = np.stack([mask_bool] * 3, axis=-1)

    color_layer = np.zeros_like(image, dtype=np.uint8)
    color_layer[:] = color

    blended_full = cv2.addWeighted(image, 1 - alpha, color_layer, alpha, 0)
    overlay[mask_3ch] = blended_full[mask_3ch]

    return overlay

def main():
    global args
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, processor = load_model(args.reasoning_model_path)

    image = cv2.imread(args.image_path)
    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    image = PILImage.fromarray(image)

    messages = prepare_message(image, args.question, args.think)
    answer_txt = process_message(model, processor, messages, device)

    try:
        think, pred_bbox, pred_points = extract_think_bbox_points(answer_txt, args.think)
        solution = json.loads(args.solution)
        iou = compute_iou(pred_bbox, solution['bbox_2d'])
        pdice = compute_pdice(pred_points, solution['point_2d'])
        
        sam2 = build_sam2("configs/sam2.1/sam2.1_hiera_t512.yaml", args.segmentation_model_path)
        seg_model = SAM2ImagePredictor(sam2)
        seg_model.set_image(image)
        masks, scores, _ = seg_model.predict(pred_points, [1,1], box=pred_bbox)
        pred_mask = masks[np.argsort(scores)[::-1][0]]
        gt_mask = np.array(mask) / 255.0
        dice = compute_dice(pred_mask, gt_mask)
    except Exception as e:
        raise ValueError(f"Failed to extract think, bbox, and points from the output: {e}")
    
    os.makedirs(args.output_path, exist_ok=True)
    im_name = os.path.basename(args.image_path)
    cv2.imwrite(os.path.join(args.output_path, im_name), np.array(image).astype(np.uint8))
    mask_name = os.path.basename(args.mask_path)
    cv2.imwrite(os.path.join(args.output_path, mask_name), np.array(pred_mask * 255.0).astype(np.uint8))
    
    vis_gt = vis_mask(np.array(image), gt_mask, color=(0, 255, 0), alpha=0.5)
    cv2.imwrite(os.path.join(args.output_path, f"gt_{im_name}"), vis_gt)
    vis_pred = vis_mask(np.array(image), pred_mask, color=(0, 0, 255), alpha=0.5)
    cv2.imwrite(os.path.join(args.output_path, f"pred_{im_name}"), vis_pred)

    result = {
        "question": args.question,
        "solution": args.solution,
        "think": think,
        "pred_bbox": pred_bbox,
        "pred_points": pred_points,
        "iou": iou,
        "pdice": pdice,
        "dice": dice,
    }    
    save_json(result, os.path.join(args.output_path, "result.json"))


if __name__ == "__main__":
    main()
