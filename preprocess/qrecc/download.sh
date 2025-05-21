wget https://github.com/apple/ml-qrecc/raw/refs/heads/main/dataset/qrecc_data.zip
mkdir -p download/qrecc
unzip qrecc_data.zip -d download/qrecc
rm qrecc_data.zip


wget https://zenodo.org/record/5115890/files/passages.zip?download=1 -O download/qrecc/passages.zip
wget https://zenodo.org/records/5543685/files/scai-qrecc21-training-turns.json?download=1 -O download/qrecc/scai-qrecc21-training-turns.json
wget https://zenodo.org/records/5543685/files/scai-qrecc21-test-turns.json?download=1 -O download/qrecc/scai-qrecc21-test-turns.json

