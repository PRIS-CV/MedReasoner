#!/bin/bash
export SWANLAB_API_KEY=w9RrLBsK15VWjl0rU6lLH
export SWANLAB_LOG_DIR=/home/dataset-assist-0/diaomuxi/yzh/swanlog

DATASET_NAME=U_MRG_14k
llamafactory-cli train pretraining/lingshu_7b_14k.yaml

# DATASET_NAME=U_MRG_6k
# llamafactory-cli train pretraining/lingshu_7b_6k.yaml

RAW_DIR=/home/dataset-assist-0/diaomuxi/model_zoo/Lingshu-7B
SRC_DIR=/home/dataset-assist-0/diaomuxi/yzh/experiments/MedReasoner/Lingshu-7B-SFT/${DATASET_NAME}
DST_DIR=/home/dataset-assist-0/diaomuxi/model_zoo/Lingshu-7B-SFT/${DATASET_NAME}

if [ -d "$DST_DIR" ]; then
    echo "[INFO] Target directory exists. Removing..."
    rm -rf "$DST_DIR"
fi

mkdir -p "$DST_DIR"
cp -r "$SRC_DIR"/README.md \
      "$SRC_DIR"/generation_config.json \
      "$SRC_DIR"/model-00001-of-00004.safetensors \
      "$SRC_DIR"/model-00002-of-00004.safetensors \
      "$SRC_DIR"/model-00003-of-00004.safetensors \
      "$SRC_DIR"/model-00004-of-00004.safetensors \
      "$SRC_DIR"/model.safetensors.index.json \
      "$SRC_DIR"/added_tokens.json \
      "$SRC_DIR"/merges.txt \
      "$SRC_DIR"/preprocessor_config.json \
      "$SRC_DIR"/special_tokens_map.json \
      "$SRC_DIR"/tokenizer.json \
      "$SRC_DIR"/tokenizer_config.json \
      "$SRC_DIR"/config.json \
      "$SRC_DIR"/video_preprocessor_config.json \
      "$SRC_DIR"/vocab.json \
      "$DST_DIR"

cp -r "$RAW_DIR"/chat_template.json "$DST_DIR"
echo "[INFO] Model files copied to $DST_DIR"
