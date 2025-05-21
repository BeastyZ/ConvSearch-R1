# this script is adapted from https://github.com/fengranMark/CHIQ/blob/main/preprocessing/preprocess_topiocqa.py
# and https://github.com/fengranMark/ConvGQR/blob/main/data/preprocess_topiocqa.py

import json
from tqdm import tqdm
import csv
import os
import jsonlines

# .tsv -> .jsonl
def convert_collection(collection_tsv, collection_json):
    with open(collection_tsv, 'r') as input, open(collection_json, 'w') as output:
        reader = csv.reader(input, delimiter="\t") # passage_nums = 25700592
        for i, row in enumerate(tqdm(reader)):
            # structure for each row: ['id', 'text', 'title'] 
            if i == 0:
                continue
            qid, text = row[0], row[1]
            title = ' '.join(row[2].split(' [SEP] '))
            obj = {"id": qid, "contents": " ".join([title, text])}
            output.write(json.dumps(obj, ensure_ascii=False) + '\n')


def gen_train_test_files(train_gold_path, dev_gold_path, train_path, dev_path, output_train, output_dev):
    with open(train_gold_path, "r") as f_gold, open(train_path, "r") as f, jsonlines.open(output_train, 'w') as fout:
        data_gold = json.load(f_gold)
        data = json.load(f)
        assert len(data_gold) == len(data)

        for d_g, d in zip(data_gold, data):
            assert d_g["conv_id"] == d["Conversation_no"]
            assert d_g["turn_id"] == d["Turn_no"]

            qid = f"topiocqa-train_{d_g['conv_id']}_{d_g['turn_id']}"
            fout.write({
                "qid": qid,
                "question": d["Question"],
                "positive_ctxs": d_g["positive_ctxs"],
                'context': d['Context'],
                'conv_id': d_g["conv_id"],
                'turn_id': d_g["turn_id"],
                'topic': d['Topic'],
                'topic_section': d['Topic_section']
            })

    with open(dev_gold_path, "r") as f_gold, open(dev_path, "r") as f, jsonlines.open(output_dev, 'w') as fout:
        data_gold = json.load(f_gold)
        data = json.load(f)
        assert len(data_gold) == len(data)

        for d_g, d in zip(data_gold, data):
            assert d_g["conv_id"] == d["Conversation_no"]
            assert d_g["turn_id"] == d["Turn_no"]

            qid = f"topiocqa-dev_{d_g['conv_id']}_{d_g['turn_id']}"
            fout.write({
                "qid": qid,
                "question": d["Question"],
                "positive_ctxs": d_g["positive_ctxs"],
                'context': d['Context'],
                'conv_id': d_g["conv_id"],
                'turn_id': d_g["turn_id"],
                'topic': d['Topic'],
                'topic_section': d['Topic_section']
            })

        
def gen_topiocqa_qrel(file_path, output_qrel_file_path, is_train=True):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    with open(output_qrel_file_path, "w") as f:
        for line in data:
            if is_train:
                sample_id = "{}_{}_{}".format("topiocqa-train", line["conv_id"], line["turn_id"])
            else:
                sample_id = "{}_{}_{}".format("topiocqa-dev", line["conv_id"], line["turn_id"])
            
            f.write("{} {} {} {}".format(sample_id, 0, line['positive_ctxs'][0]['passage_id'], 1))
            f.write('\n')


if __name__ == "__main__":
    collection_tsv = "download/downloads/data/wikipedia_split/full_wiki_segments.tsv"
    collection_json = "data/topiocqa/collection/full_wiki_segments.jsonl"
    dirname = os.path.dirname(collection_json)
    os.makedirs(dirname, exist_ok=True)
    convert_collection(collection_tsv, collection_json)

    train_gold = "download/downloads/data/retriever/all_history/train.json"
    dev_gold = "download/downloads/data/retriever/all_history/dev.json"
    train = 'download/downloads/data/topiocqa_dataset/train.json'
    dev = 'download/downloads/data/topiocqa_dataset/dev.json'
    output_train = "data/topiocqa/train.json"
    output_dev = "data/topiocqa/dev.json"
    gen_train_test_files(train_gold, dev_gold, train, dev, output_train, output_dev)

    train_trec = "data/topiocqa/train.trec"
    dev_trec = "data/topiocqa/dev.trec"
    gen_topiocqa_qrel(train_gold, train_trec, True)
    gen_topiocqa_qrel(dev_gold, dev_trec, False)
