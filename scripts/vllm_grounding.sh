#!/bin/bash

DATASET_NAME=U_MRG_14k
# DATASET_NAME=U_MRG_6k
DATA_PATH=/home/dataset-assist-0/diaomuxi/dataset_zoo/${DATASET_NAME}

# ======================
# Stage 1: Run Inference
# ======================

# Close-Source Models
# export API_BASE='http://38.97.60.202:9088/v1'
# MODEL_NAME=GPT-4o
# export API_MODE='gpt-4o'
# export API_KEY='YOUR-API-KEY'
# MODEL_NAME=Gemini-2.5-Flash
# export API_MODE='gemini-2.5-flash'
# export API_KEY='YOUR-API-KEY'

export API_BASE='http://localhost:18900/v1'
# VLM models
MODEL_NAME=Qwen2.5-VL-72B-Instruct
# MODEL_NAME=Qwen2.5-VL-7B-Instruct
# MODEL_NAME=InternVL3-78B-Instruct
# MODEL_NAME=InternVL3-8B-Instruct
# MODEL_NAME=MedVLM-R1
# MODEL_NAME=medgemma-4b-it
# MODEL_NAME=Mini-InternVL2-4B-DA-Medical
# MODEL_NAME=HuatuoGPT-Vision-7B-Qwen2.5VL
# MODEL_NAME=Lingshu-7B
# MODEL_NAME=Chiron-o1-8B

# VLM models with SFT
# MODEL_NAME=Lingshu-7B-SFT

# VLM models with GRPO
# MODEL_NAME=Lingshu-7B-Base
# MODEL_NAME=Lingshu-7B-Hard
# MODEL_NAME=Lingshu-7B-Soft

for THINK in True False; do
    echo "[INFO] Running inference with THINK=${THINK}"

    if [ "$THINK" = "True" ]; then
        OUTPUT_PATH="/home/dataset-assist-0/diaomuxi/yzh/experiments/${DATASET_NAME}/${MODEL_NAME}_Think"
    else
        OUTPUT_PATH="/home/dataset-assist-0/diaomuxi/yzh/experiments/${DATASET_NAME}/${MODEL_NAME}_Base"
    fi

    mkdir -p "$OUTPUT_PATH"

    python eval/vllm_eval.py \
        --model_name "${MODEL_NAME}" \
        --think "${THINK}" \
        --output_path "${OUTPUT_PATH}" \
        --data_path "${DATA_PATH}"

    echo "[INFO] All inference processes for THINK=${THINK} completed."
    echo
done

echo "[INFO] All THINK modes inference completed."

# ======================
# Stage 2: Run Statistics
# ======================

SEG_MODEL_PATHS=(
    /home/dataset-assist-0/diaomuxi/model_zoo/medsam-vit-base/medsam_vit_b.pth
    /home/dataset-assist-0/diaomuxi/model_zoo/SAM-Med2D_model/sam-med2d_b.pth
    /home/dataset-assist-0/diaomuxi/model_zoo/MedSAM2/MedSAM2_latest.pt
)

GPU_IDS=(0 1 2)

for i in "${!SEG_MODEL_PATHS[@]}"; do
    SEGMENTATION_MODEL_PATH="${SEG_MODEL_PATHS[$i]}"
    GPU_ID="${GPU_IDS[$i]}"

    echo "[INFO] Launching statistics on GPU ${GPU_ID} with SEG_MODEL=${SEGMENTATION_MODEL_PATH}"

    {
        for THINK in True False; do
            if [ "$THINK" = "True" ]; then
                OUTPUT_PATH="/home/dataset-assist-0/diaomuxi/yzh/experiments/${DATASET_NAME}/${MODEL_NAME}_Think"
            else
                OUTPUT_PATH="/home/dataset-assist-0/diaomuxi/yzh/experiments/${DATASET_NAME}/${MODEL_NAME}_Base"
            fi

            INPUT_JSON="${OUTPUT_PATH}/results.json"
            OUTPUT_JSON="${OUTPUT_PATH}/statistics.json"

            echo "[INFO][GPU $GPU_ID] Starting statistics for THINK=$THINK"

            CUDA_VISIBLE_DEVICES=$GPU_ID python eval/statistic.py \
                --segmentation_model_path "${SEGMENTATION_MODEL_PATH}" \
                --data_path "${DATA_PATH}" \
                --input_json "${INPUT_JSON}" \
                --output_json "${OUTPUT_JSON}"

            echo "[INFO][GPU $GPU_ID] Saved statistics to ${OUTPUT_JSON}"
        done
    } &
done

wait
echo "[INFO] All statistics completed."
