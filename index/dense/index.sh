export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# for topiocqa
# python3 index/dense/dense_index.py \
#     --pretrained_doc_encoder_path PATH/TO/YOUR/ANCE \
#     --output_index_dir_path embedding/ance_topiocqa \
#     --collection_path data/topiocqa/collection/full_wiki_segments.jsonl \
#     --dataset topiocqa \
#     --per_gpu_index_batch_size 8192


# for qrecc
python3 index/dense/dense_index.py \
    --pretrained_doc_encoder_path PATH/TO/YOUR/ANCE \
    --output_index_dir_path embedding/ance_qrecc \
    --collection_path data/qrecc/collection/qrecc_collection.jsonl \
    --dataset qrecc \
    --per_gpu_index_batch_size 8192
