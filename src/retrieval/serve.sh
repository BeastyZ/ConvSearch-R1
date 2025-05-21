# followling code is optional
# sudo mkdir -p /usr/lib/jvm
# sudo tar -xzf jdk-21_linux-x64_bin.tar.gz -C /usr/lib/jvm/
# export JAVA_HOME=/usr/lib/jvm/jdk-21.0.6
# export PATH=$JAVA_HOME/bin:$PATH
# java -version

conda activate retriever

# for topiocqa (sparse)
nohup python3 src/retrieval/server.py \
    --index_path embedding/bm25_topiocqa \
    --trec_path data/topiocqa/train.trec \
    --retriever_name bm25 \
    --port $PORT0 \
    --bm25_k1 0.9 \
    --bm25_b 0.4 > logs/topiocqa_sparse.log 2>&1 &

# for qrecc (sparse)
nohup python3 src/retrieval/server.py \
    --index_path embedding/bm25_qrecc \
    --trec_path data/qrecc/train.trec \
    --retriever_name bm25 \
    --port $PORT1 \
    --bm25_k1 0.82 \
    --bm25_b 0.68 > logs/qrecc_sparse.log 2>&1 &

# for topiocqa (dense)
nohup python3 src/retrieval/server.py \
    --index_path embedding/ance_topiocqa/doc_emb_cat.pkl \
    --trec_path data/topiocqa/train.trec \
    --corpus_docid_path embedding/ance_topiocqa/doc_embid_cat.pkl \
    --retriever_name ance \
    --model_name_or_path path/to/your/dense/retriever \
    --faiss_gpu \
    --n_gpu 4 \
    --port $PORT2 > logs/topiocqa_dense.log 2>&1 &

# for qrecc (dense)
python3 src/retrieval/server.py \
    --index_path embedding/ance_qrecc/doc_emb_cat.pkl \
    --trec_path data/qrecc/train.trec \
    --corpus_docid_path embedding/ance_qrecc/doc_embid_cat.pkl \
    --retriever_name ance \
    --model_name_or_path path/to/your/dense/retriever \
    --faiss_gpu \
    --n_gpu 4 \
    --port $PORT3 > logs/qrecc_dense.log 2>&1 &

echo "Port0 for topiocqa sparse: $PORT0"
echo "Port1 for qrecc sparse: $PORT1"
echo "Port2 for topiocqa dense: $PORT2"
echo "Port3 for qrecc dense: $PORT3"

