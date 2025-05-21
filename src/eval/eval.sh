
# dense eval for topiocqa
python3 src/eval/get_metrics_using_ance.py \
    --pretrained_encoder_path path/to/your/dense/retriever \
    --test_file_path path/to/rewrite/file/gennerated/by/rewriter \
    --passage_embeddings_dir_path "embedding/ance_topiocqa" \
    --qrel_output_dir data/topiocqa/qrel/dense \
    --output_trec_file filename/to/save/trec/file \
    --trec_gold_qrel_file_path data/topiocqa/dev.trec \
    --n_gpu 2 \
    --test_type rewrite


# dense eval for qrecc
python3 src/eval/get_metrics_using_ance.py \
    --pretrained_encoder_path path/to/your/dense/retriever \
    --test_file_path path/to/rewrite/file/gennerated/by/rewriter \
    --passage_embeddings_dir_path "embedding/ance_qrecc" \
    --qrel_output_dir data/qrecc/qrel/dense \
    --output_trec_file filename/to/save/trec/file \
    --trec_gold_qrel_file_path data/qrecc/test.trec \
    --n_gpu 4 \
    --test_type rewrite


# sparse retrieval
# this operation is optional
# sudo mkdir -p /usr/lib/jvm
# sudo tar -xzf jdk-21_linux-x64_bin.tar.gz -C /usr/lib/jvm/
# export JAVA_HOME=/usr/lib/jvm/jdk-21.0.6
# export PATH=$JAVA_HOME/bin:$PATH
# java -version


# sparse eval for topiocqa
python3 src/eval/get_metrics_using_bm25.py \
    --index_dir_path embedding/bm25_topiocqa \
    --query_file path/to/rewrite/file/gennerated/by/rewriter \
    --gold_qrel_file_path data/topiocqa/dev.trec \
    --bm25_k1 0.9 \
    --bm25_b 0.4 \
    --retrieval_output_path data/topiocqa/qrel/bm25 \
    --output_file_name filename/to/save/trec/file


# sparse eval for qrecc
python3 src/eval/get_metrics_using_bm25.py \
    --index_dir_path embedding/bm25_qrecc \
    --query_file path/to/rewrite/file/gennerated/by/rewriter \
    --gold_qrel_file_path data/qrecc/test.trec \
    --bm25_k1 0.82 \
    --bm25_b 0.68 \
    --retrieval_output_path data/qrecc/qrel/bm25 \
    --output_file_name filename/to/save/trec/file

