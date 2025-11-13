# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
)


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement) -> torch.Tensor:
    """Merge sharded tensors based on placement strategy."""
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")

def find_total_shards(local_dir: str) -> str:
    """Extract world size and validate file format."""
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            return match.group(1)
    raise ValueError("No model file with the expected naming convention found.")

def load_model_shards(local_dir: str, world_size: int, total_shards: int) -> List[Dict]:
    """Parallel loading of all model shards."""
    model_state_dicts = [None] * total_shards
    def load_shard(rank):
        path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        model_state_dicts[rank] = torch.load(path, map_location="cpu", weights_only=False)
    load_shard(0)  # Load rank 0 first
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        executor.map(load_shard, range(1, total_shards))
    return model_state_dicts

def merge_shards(model_state_dicts: List[Dict], mesh_dim_names: Tuple[str]) -> Dict:
    """Merge model weights across shards."""
    param_placements: Dict[str, List[Placement]] = {}
    merged_state_dict = {}
    keys = list(model_state_dicts[0].keys())

    for key in keys:
        shard_tensors = []
        for shard in model_state_dicts:
            tensor = shard.pop(key)
            if isinstance(tensor, DTensor):
                shard_tensors.append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                if mesh_dim_names[0] == "dp":
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                merged_state_dict[key] = tensor.bfloat16()
        if shard_tensors:
            placement = param_placements[key][0]
            merged_state_dict[key] = merge_by_placement(shard_tensors, placement)

    return merged_state_dict

def infer_model_class(config) -> torch.nn.Module:
    """Determine the correct model class from config."""
    arch = config.architectures[0]
    if "ForTokenClassification" in arch:
        return AutoModelForTokenClassification
    elif "ForCausalLM" in arch:
        return AutoModelForCausalLM
    elif "ForConditionalGeneration" in arch:
        return AutoModelForVision2Seq
    else:
        raise NotImplementedError(f"Unknown architecture: {arch}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=str, help="The path for your saved model")
    parser.add_argument("--save_dir", required=True, type=str, help="The path to save the merged model")
    parser.add_argument("--hf_path", default=False, type=str, help="The path of the huggingface repo to upload")
    args = parser.parse_args()

    assert not args.local_dir.endswith("huggingface"), "The local_dir should not end with huggingface"
    local_dir = args.local_dir

    # === Step 1: Get world_size string (e.g., '8') from filenames ===
    world_size = find_total_shards(local_dir)

    # === Step 2: Load rank 0 to get mesh info ===
    state_dict_0 = torch.load(os.path.join(local_dir, f"model_world_size_{world_size}_rank_0.pt"), map_location="cpu")
    pivot_key = sorted(state_dict_0.keys())[0]
    weight = state_dict_0[pivot_key]
    assert isinstance(weight, DTensor)

    device_mesh = weight.device_mesh
    mesh = device_mesh.mesh
    mesh_dim_names = device_mesh.mesh_dim_names

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")
    assert mesh_dim_names in (("fsdp",),), f"Unsupported mesh_dim_names {mesh_dim_names}"

    # === Step 3: Determine total_shards ===
    if "tp" in mesh_dim_names:
        total_shards = mesh.shape[-1] * mesh.shape[-2]
    else:
        total_shards = mesh.shape[-1]

    print(f"Processing model shards with total_shards={total_shards}")

    # === Step 4: Load all shards ===
    model_state_dicts = load_model_shards(local_dir, world_size, total_shards)

    # === Step 5: Merge weights ===
    merged_state_dict = merge_shards(model_state_dicts, mesh_dim_names)

    # === Step 6: Load config & create model ===
    config = AutoConfig.from_pretrained(os.path.join(local_dir, "huggingface"))
    model_cls = infer_model_class(config)

    with torch.device("meta"):
        model = model_cls.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device="cpu")

    print(f"Saving model to {args.save_dir}")
    model.save_pretrained(args.save_dir, state_dict=merged_state_dict)

    del merged_state_dict
    del model
    print("Model merging completed.")

    # === Step 7: Optionally upload to Hugging Face ===
    if args.hf_path:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.hf_path, private=False, exist_ok=True)
        api.upload_folder(folder_path=args.save_dir, repo_id=args.hf_path, repo_type="model")
        print(f"Model uploaded to HuggingFace Hub: {args.hf_path}")

if __name__ == "__main__":
    main()
