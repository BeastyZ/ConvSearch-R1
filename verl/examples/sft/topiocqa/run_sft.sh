set -x

nproc_per_node=8
save_path=ckpt/sft/topiocqa/qwen2.5-3b-it_self
export WANDB_API_KEY="your_wandb_key"
HOME=path/to/your/work/home

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/topiocqa/sft/train_qwen2.5-3b-it_self.parquet \
    data.val_files=$HOME/data/topiocqa/sft/test_qwen2.5-3b-it_self.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['prompt'] \
    +data.response_dict_keys=['answer'] \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=8 \
    data.max_length=3072 \
    model.partial_pretrain=path/to/your/model \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=qwen2.5-3b-it_topiocqa-sft \
    trainer.experiment_name=qwen2.5-3b-it_self_epoch2 \
    trainer.total_epochs=2 \
    trainer.logger=['wandb'] \
    trainer.default_hdfs_dir=null


save_path=ckpt/sft/topiocqa/llama3.2-3b-it_self

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/topiocqa/sft/train_llama3.2-3b-it_self.parquet \
    data.val_files=$HOME/data/topiocqa/sft/test_llama3.2-3b-it_self.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['prompt'] \
    +data.response_dict_keys=['answer'] \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=8 \
    data.max_length=3072 \
    model.partial_pretrain=path/to/your/model \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=llama3.2-3b-it_topiocqa-sft \
    trainer.experiment_name=llama3.2-3b-it_self_epoch2 \
    trainer.total_epochs=2 \
    trainer.logger=['wandb'] \
    trainer.default_hdfs_dir=null

