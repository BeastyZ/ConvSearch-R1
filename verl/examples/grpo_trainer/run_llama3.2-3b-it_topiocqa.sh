#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY="your_wandb_key"
DATE=$(date "+%y%m%d%H%M")
HOME=path/to/your/work/home

EXPERIMENT_NAME=llama3.2_3b_it_self_bs128_maxlen1024_lr1e-6_warmup100_n16_temp0.7_epoch6_r9
export RETRIEVER_URL="your_retrieval_server_url"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/topiocqa/train.parquet \
    data.val_files=$HOME/data/topiocqa/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1536 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=path/to/your/sft/model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=100 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='verl_grpo_rewrite_topiocaqa' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.default_local_dir=ckpt/topiocqa/dense/$EXPERIMENT_NAME \
    trainer.total_epochs=6 \
    reward_model.reward_manager=rewrite_r1 \
    custom_reward_function.path=verl/verl/utils/reward_score/rewrite_r1.py \
    retriever.topk=100 $@ 2>&1 | tee logs/${DATE}_${EXPERIMENT_NAME}.log
