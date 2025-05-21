# collection
python3 preprocess/topiocqa/download_data.py \
    --resource data.wikipedia_split.full_wiki_segments \
    --output_dir download


# train
python3 preprocess/topiocqa/download_data.py \
    --resource data.topiocqa_dataset.train \
    --output_dir download


# dev
python3 preprocess/topiocqa/download_data.py \
    --resource data.topiocqa_dataset.dev \
    --output_dir download


# gold passages train
python3 preprocess/topiocqa/download_data.py \
    --resource data.retriever.all_history.train \
    --output_dir download


# gold passages dev
python3 preprocess/topiocqa/download_data.py \
    --resource data.retriever.all_history.dev \
    --output_dir download
