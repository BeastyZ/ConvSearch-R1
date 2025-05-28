# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# for qrecc
torchrun --nproc_per_node=8 --master_port=12355 index/dense/distributed_dense_index.py \
    --pretrained_doc_encoder_path PATH/TO/YOUR/ANCE \
    --output_index_dir_path embedding/ance_qrecc/dist \
    --collection_path data/qrecc/collection/qrecc_collection.jsonl \
    --per_gpu_index_batch_size 1024
