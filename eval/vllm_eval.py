import argparse
import os
import json
import asyncio
import base64
from io import BytesIO
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from datasets import load_dataset

from utils import extract_think_bbox_points, compute_iou, compute_pdice, save_json


MAX_CONCURRENT_REQUESTS = 64
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

API_BASE = os.environ.get('API_BASE', "http://localhost:18900/v1")
API_KEY = os.environ.get('API_KEY', "EMPTY")
API_MODE = os.environ.get('API_MODE', None)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of the model to use for evaluation.")
    parser.add_argument("--think", type=str, help="Whether to include thinking in the output (True/False).")
    parser.add_argument("--output_path", type=str, help="Path to save the output results.")
    parser.add_argument("--data_path", type=str, help="Path to the test dataset.")
    return parser.parse_args()

def encode_pil_image_to_base64(pil_image, format="PNG"):
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str, format.lower()

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
            "{Question}\n\n"
            
            "Output Formats:\n"
            "Your response must adhere to a strict format, containing exactly one <answer> block:\n"
            "- <answer>...</answer>: This section must contain a JSON object with the following keys and values:\n"
            "    \"bbox\": the tightest bounding box enclosing the target region.\n"
            "    \"points_1\": a primary key point within the bbox, on the target region.\n"
            "    \"points_2\": a second, distinct key point within the target region.\n\n"
            
            "Example Output:\n"
            "<answer>{Answer}</answer>"
        )
    
    system_message = { "role": "system", "content": "You are a helpful assistant." }

    base64_image, image_type = encode_pil_image_to_base64(item["images"][0])
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_type};base64,{base64_image}"
                }
            },
            {
                "type": "text",
                "text": user_prompt.format(
                    Question=item['prompt'][0]['content'],
                    Answer="{'bbox': [xmin, ymin, xmax, ymax], 'points_1': [x1, y1], 'points_2': [x2, y2]}"
                )
            }
        ]
    }

    message = [system_message, user_message]
    return message

async def process_response(client, user_message, retries=5, timeout=60):
    for attempt in range(retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=API_MODE,
                    messages=user_message,
                    timeout=timeout,
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(1.5 * (attempt + 1))

async def normalize_coordinates(bbox, points, norm_size=1000):
    bbox = [round(coord / norm_size * 840) for coord in bbox]
    points = [[round(x / norm_size * 840), round(y / norm_size * 840)] for x, y in points]

    return bbox, points

async def process_item(client, item, model_name, think):
    try:
        user_message = prepare_message(item, think)
        response = await process_response(client, user_message)
        
        think, bbox, points = extract_think_bbox_points(response, think)
        # if model_name.startswith("InternVL3"):
        #     bbox, points = await normalize_coordinates(bbox, points, norm_size=1000)

        solution = json.loads(item['reward_model']['ground_truth'])
        iou = compute_iou(bbox, solution['bbox_2d'])
        pdice = compute_pdice(points, solution['point_2d'])
    except Exception as e:
        print(f"Error processing item {item['extra_info']['id']}: {e}")
        think, bbox, points, iou, pdice = "error", None, None, 0.0, 0.0
        
    result = {
        "id": item["extra_info"]["id"],
        "problem": item['prompt'][0]['content'],
        "supercategory": item["extra_info"]["supercategory"],
        "category": item["extra_info"]["category"],
        "output": response,
        "think": think,
        "bbox": bbox,
        "points": points,
        "iou": round(iou, 4),
        "pdice": round(pdice, 4),
    }
    return result
    
async def process_dataset(client, dataset, model_name, think):
    tasks = []
    for item in dataset:
        tasks.append(process_item(client, item, model_name, think))
    
    results = await tqdm_asyncio.gather(*tasks, desc="Eval Dataset")
    return results

async def main(args, dataset):
    client = AsyncOpenAI(
        base_url=API_BASE,
        api_key=API_KEY
    )

    print(f"Loaded {len(dataset)} items from the dataset.") 
    
    results = await process_dataset(client, dataset, args.model_name, args.think)
    save_json(results, os.path.join(args.output_path, f"results.json"))

if __name__ == "__main__":
    args = parse_args()
    dataset = load_dataset(args.data_path)['test']
    asyncio.run(main(args, dataset))
    