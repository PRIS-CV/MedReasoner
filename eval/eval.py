import os
import json
import sys
import argparse
from tqdm import tqdm

import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info

from utils import extract_think_bbox_points, compute_iou, compute_pdice, save_json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, help="Path to the reasoning model.")
    parser.add_argument("--think", type=str, help="Whether to include thinking in the output (True/False).", default="True")
    parser.add_argument("--data_path", type=str, help="Path to the test dataset.")
    parser.add_argument("--output_path", type=str, help="Path to save the output results.")
    parser.add_argument("--idx", type=int, help="Index for the current part of the dataset.")
    parser.add_argument("--num_parts", type=int, help="Total number of parts to split the dataset into.")
    parser.add_argument("--batch_size", type=int, default=50)
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

def prepare_message(item, think="True"):
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
            {"type": "image", "image": item["images"][0]},
            {"type": "text", "text": user_prompt.format
                (
                    Question=item['prompt'][0]['content'],
                    Answer="{'bbox': [xmin, ymin, xmax, ymax], 'points_1': [x1, y1], 'points_2': [x2, y2]}"
                )
            }
        ]}
    ]

def process_message(model, processor, batch_msg, device):
    text = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_msg]
    image_inputs, video_inputs = process_vision_info(batch_msg)

    inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, use_cache=True, max_new_tokens=2048, do_sample=False)
    decoded = processor.batch_decode([out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)],
                                        skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return decoded

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset(args.data_path)['test']
    total_len = len(dataset)
    start_idx = args.idx * (total_len // args.num_parts)
    end_idx = total_len if args.idx == args.num_parts - 1 else (args.idx + 1) * (total_len // args.num_parts)
    dataset = dataset.select(range(start_idx, end_idx))
    
    model, processor = load_model(args.reasoning_model_path)

    messages, id_list, results = [], [], []
    for item in dataset:
        messages.append(prepare_message(item, args.think))
        id_list.append({
            "id": item["extra_info"]["id"],
            "problem": item['prompt'][0]['content'],
            "solution": item['reward_model']['ground_truth'],
            "image": item["images"][0],
            "mask": item["masks"][0],
            "extra_info": item["extra_info"]
        })

    for i in tqdm(range(0, len(messages), args.batch_size)):
        batch_msg, batch_id = messages[i:i+args.batch_size], id_list[i:i+args.batch_size]
        decoded = process_message(model, processor, batch_msg, device)
        
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            for j, out_text in enumerate(decoded):
                try:
                    think, bbox, points = extract_think_bbox_points(out_text, args.think)
                    solution = json.loads(batch_id[j]['solution'])
                    iou = compute_iou(bbox, solution['bbox_2d'])
                    pdice = compute_pdice(points, solution['point_2d'])
                except Exception as e:
                    print(f"[!] Error for ID {batch_id[j]['id']}: {e}")
                    print(f"[!] Output text: {out_text}")
                    think, bbox, points, iou, pdice = "error", None, None, 0.0, 0.0

                results.append({
                    "id": batch_id[j]["id"],
                    "problem": batch_id[j]["problem"],
                    "supercategory": batch_id[j]["extra_info"]["supercategory"],
                    "category": batch_id[j]["extra_info"]["category"],
                    "output": out_text,
                    "think": think,
                    "bbox": bbox,
                    "points": points,
                    "iou": round(iou, 4),
                    "pdice": round(pdice, 4),
                })

        torch.cuda.empty_cache()

    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    save_json(results, output_file)


if __name__ == "__main__":
    main()
