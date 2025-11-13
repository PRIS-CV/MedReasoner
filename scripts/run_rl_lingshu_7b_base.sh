#!/bin/bash

set -x

DATASET_NAME=U_MRG_14k
# DATASET_NAME=U_MRG_6k
TRAIN_FILES=/home/dataset-assist-0/diaomuxi/dataset_zoo/${DATASET_NAME}/train.parquet
VAL_FILES=/home/dataset-assist-0/diaomuxi/dataset_zoo/${DATASET_NAME}/test.parquet

MODEL_PATH=/home/dataset-assist-0/diaomuxi/model_zoo/Lingshu-7B

OUTPUT_PATH=/home/dataset-assist-0/diaomuxi/yzh/experiments
PROJECT_NAME=MedReasoner
EXPERIMENT_NAME=Lingshu-7B-Base
CKPT_SAVE_PATH=${OUTPUT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME}/${DATASET_NAME}

# Please first set reward_mode = 'base' in line 87 of verl/utils/reward_score/__init__.py

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.image_key=images \
    data.custom_cls.name=UMRG \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward_model.reward_manager=med_reasoner \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    trainer.total_epochs=5 \
    trainer.save_freq=10 \
    trainer.test_freq=1 \
    trainer.default_local_dir=$CKPT_SAVE_PATH \
