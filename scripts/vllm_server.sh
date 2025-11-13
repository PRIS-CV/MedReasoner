#!/bin/bash

# VLM models
MODEL_NAME=Qwen2.5-VL-72B-Instruct
# MODEL_NAME=InternVL3-78B-Instruct
PARALLEL_SIZE=8
# MODEL_NAME=Qwen2.5-VL-7B-Instruct
# MODEL_NAME=InternVL3-8B-Instruct
# MODEL_NAME=MedVLM-R1
# MODEL_NAME=medgemma-4b-it
# MODEL_NAME=Mini-InternVL2-4B-DA-Medical
# MODEL_NAME=HuatuoGPT-Vision-7B-Qwen2.5VL
# MODEL_NAME=Lingshu-7B
# MODEL_NAME=Chiron-o1-8B
# PARALLEL_SIZE=4
REASONING_MODEL_PATH=/home/dataset-assist-0/diaomuxi/model_zoo/${MODEL_NAME}

DATASET_NAME=U_MRG_14k
# DATASET_NAME=U_MRG_6k
# PARALLEL_SIZE=4

# VLM models with SFT
# MODEL_NAME=Lingshu-7B-SFT
# REASONING_MODEL_PATH=/home/dataset-assist-0/diaomuxi/model_zoo/${MODEL_NAME}/${DATASET_NAME}

# VLM models with GRPO
# MODEL_NAME=Lingshu-7B-Base
# MODEL_NAME=Lingshu-7B-Hard
# MODEL_NAME=Lingshu-7B-Soft
# REASONING_MODEL_PATH=/home/dataset-assist-0/diaomuxi/model_zoo/${MODEL_NAME}/${DATASET_NAME}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve ${REASONING_MODEL_PATH} \
    --port 18900 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size ${PARALLEL_SIZE} \
    --served-model-name "vllm_eval" \
    --trust-remote-code \
    --limit-mm-per-prompt image=5 \
