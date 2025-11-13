#!/bin/bash

GT_PATH=/home/dataset-assist-0/diaomuxi/dataset_zoo/UMRGv1/test

PRED_PATH=/home/dataset-assist-0/diaomuxi/yzh/experiments/U_MRG_6k/Lingshu-7B-Soft_Think

ID_LIST=covid_1583_X-Ray_chest_left_lung_000.png

python eval/vis_results.py \
    --id_list ${ID_LIST} \
    --gt_json ${GT_PATH}/umrg_rl.json \
    --pred_json ${PRED_PATH}/results_epoch230.json \
    --image_root ${GT_PATH} \
    --output_dir ${PRED_PATH}/vis_results \
