#!/bin/bash

BASE_PATH=/home/dataset-assist-0/diaomuxi/yzh/experiments/MedReasoner

DATASET_NAME=U_MRG_14k
# DATASET_NAME=U_MRG_6k

# MODEL_NAME=Lingshu-7B-Base
# MODEL_NAME=Lingshu-7B-Hard
MODEL_NAME=Lingshu-7B-Soft

MODEL_PATH=${BASE_PATH}/${MODEL_NAME}/${DATASET_NAME}
mkdir -p ${MODEL_PATH}/merged_model

STEP=global_step_480

python eval/model_merger.py \
    --local_dir ${MODEL_PATH}/${STEP}/actor \
    --save_dir ${MODEL_PATH}/merged_model

sudo chown -R batchcom:batchcom ${MODEL_PATH}
shopt -s dotglob
cp -r "${MODEL_PATH}/${STEP}/actor/huggingface/"* "${MODEL_PATH}/merged_model/"
shopt -u dotglob

DEST_PATH=/home/dataset-assist-0/diaomuxi/model_zoo/${MODEL_NAME}/${DATASET_NAME}
mkdir -p ${DEST_PATH}

if [ -d "${DEST_PATH}" ]; then
    echo "[INFO] Deleting existing directory: ${DEST_PATH}"
    rm -rf "${DEST_PATH}"
fi

mv "${MODEL_PATH}/merged_model" "${DEST_PATH}"
echo "[INFO] Merged model files moved to ${DEST_PATH}"
