export VLLM_USE_V1=1

# for ConvSearch-R1 (topiocqa)
python3 src/infer/infer.py \
    --model_name_or_path path/to/your/model \
    --model_name ConvSearch-R1 \
    --dp_size 8 \
    --gpus_per_dp_rank 1 \
    --temperature 0.7 \
    --input_path data/topiocqa/dev.json \
    --output_path path/to/your/output


# for ConvSearch-R1 (qrecc)
python3 src/infer/infer.py \
    --model_name_or_path path/to/your/model \
    --model_name ConvSearch-R1 \
    --dp_size 8 \
    --gpus_per_dp_rank 1 \
    --temperature 0.7 \
    --input_path data/qrecc/test_v3.json \
    --output_path path/to/your/output


# for getting SFT data (topiocqa)
python3 src/infer/infer.py \
    --model_name_or_path path/to/your/model \
    --model_name Get_SFT \
    --dp_size 8 \
    --gpus_per_dp_rank 1 \
    --temperature 0.7 \
    --input_path data/topiocqa/train.json \
    --output_path path/to/your/output


# for getting SFT data (qrecc)
python3 src/infer/infer.py \
    --model_name_or_path path/to/your/model \
    --model_name Get_SFT \
    --dp_size 8 \
    --gpus_per_dp_rank 1 \
    --temperature 0.7 \
    --input_path data/qrecc/train_v3.json \
    --output_path path/to/your/output
