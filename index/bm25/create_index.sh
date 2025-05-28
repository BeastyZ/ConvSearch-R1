# sudo tar -xzf jdk-21_linux-x64_bin.tar.gz -C /usr/lib/jvm/

# export JAVA_HOME=/usr/lib/jvm/jdk-21.0.6
# export PATH=$JAVA_HOME/bin:$PATH
java -version

OUTPUT=embedding/bm25_qrecc
INPUT=data/qrecc/collection/

if [ ! -f "$OUTPUT" ]; then
    echo "Creating index..."
    python -m pyserini.index -collection JsonCollection \
                            -generator DefaultLuceneDocumentGenerator \
                            -threads 20 \
                            -input ${INPUT} \
                            -index ${OUTPUT} \
							-storeRaw
fi
