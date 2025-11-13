#!/bin/bash

DATASET_NAME=U_MRG_14k
# DATASET_NAME=U_MRG_6k

# VLM models
# MODEL_NAME=Qwen2.5-VL-72B-Instruct
# MODEL_NAME=Qwen2.5-VL-7B-Instruct
# MODEL_NAME=InternVL3-78B-Instruct
# MODEL_NAME=InternVL3-8B-Instruct
# MODEL_NAME=MedVLM-R1
# MODEL_NAME=medgemma-4b-it
# MODEL_NAME=HuatuoGPT-Vision-7B-Qwen2.5VL
# MODEL_NAME=Lingshu-7B
# REASONING_MODEL_PATH=/home/dataset-assist-0/diaomuxi/model_zoo/${MODEL_NAME}

# VLM models with SFT
# MODEL_NAME=Lingshu-7B-SFT
# REASONING_MODEL_PATH=/home/dataset-assist-0/diaomuxi/model_zoo/${MODEL_NAME}/${DATASET_NAME}

# VLM models with RL
# MODEL_NAME=Lingshu-7B-Base
# MODEL_NAME=Lingshu-7B-Hard
MODEL_NAME=Lingshu-7B-Soft
REASONING_MODEL_PATH=/home/dataset-assist-0/diaomuxi/model_zoo/${MODEL_NAME}/${DATASET_NAME}

SEGMENTATION_MODEL_PATH=/home/dataset-assist-0/diaomuxi/model_zoo/MedSAM2/MedSAM2_latest.pt

# Create output directory
THINK=True
# THINK=False
if [ "$THINK" = "True" ]; then
    OUTPUT_PATH="/home/dataset-assist-0/diaomuxi/yzh/experiments/${DATASET_NAME}/${MODEL_NAME}_Think/infer"
else
    OUTPUT_PATH="/home/dataset-assist-0/diaomuxi/yzh/experiments/${DATASET_NAME}/${MODEL_NAME}_Base/infer"
fi
mkdir -p $OUTPUT_PATH

DATA_PATH=/home/dataset-assist-0/diaomuxi/dataset_zoo/UMRGv1
IMAGE_PATH="${DATA_PATH}/test/images/covid_1583_X-Ray_chest.png"
MASK_PATH="${DATA_PATH}/test/masks/covid_1583_X-Ray_chest_left_lung_000.png"
QUESTION="What can be observed in the structure occupying the left side, marked by an elongated shadow and branching features?"
SOLUTION='{"bbox_2d": [445, 136, 708, 729], "point_2d": [[534, 273], [595, 455]]}'

python eval/infer.py \
    --reasoning_model_path ${REASONING_MODEL_PATH} \
    --segmentation_model_path ${SEGMENTATION_MODEL_PATH} \
    --think ${THINK} \
    --image_path ${IMAGE_PATH} \
    --mask_path ${MASK_PATH} \
    --question "${QUESTION}" \
    --solution "${SOLUTION}" \
    --output_path ${OUTPUT_PATH}
