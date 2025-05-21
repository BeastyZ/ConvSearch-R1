# this code is adapted from https://github.com/fengranMark/CHIQ/blob/main/preprocessing/preprocess_qrecc.py

import json
import os
from tqdm import tqdm
import logging
import pickle
import copy
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))

def gen_qrecc_passage_collection(input_passage_path, output_file, pid2rawpid_path):
    """
    structure of qrecc passages.zip:
    - passages.zip
        - collection-paragraph
            - wayback
                - *.jsonl
            - commoncrawl
                - *.jsonl
            - wayback-backfill
                - *.jsonl
    """
    def process_qrecc_per_dir(file_paths, pid, pid2rawpid, fw, zip_file):
        for path in tqdm(file_paths):
            with zip_file.open(path) as file:
                for line in file:
                    line = json.loads(line.decode('utf-8'))
                    raw_pid = line["id"]
                    passage = line["contents"]
                    pid2rawpid[str(pid)] = raw_pid
                    # fw.write("{}\t{}".format(pid, passage))
                    fw.write(json.dumps({"id": str(pid), "contents": passage}))
                    fw.write("\n")
                    pid += 1
        
        return pid, pid2rawpid

    import zipfile
    """Processing sequence:
        commoncrawl -> wayback -> wayback-backfill
    """

    with zipfile.ZipFile(input_passage_path, 'r') as zip_file:
        all_files = zip_file.namelist()

        wayback_paths = []
        wayback_backfill_paths = []
        commoncrawl_paths = []
        for file_path in all_files:
            if 'wayback' in file_path and file_path.endswith('.jsonl'):
                wayback_paths.append(file_path)
            elif 'wayback-backfill' in file_path and file_path.endswith('.jsonl'):
                wayback_backfill_paths.append(file_path)
            elif 'commoncrawl' in file_path and file_path.endswith('.jsonl'):
                commoncrawl_paths.append(file_path)
            else:
                print(f'{file_path} is not a json file')

        pid = 0
        pid2rawpid = {}
        with open(output_file, "w") as fw:
            pid, pid2rawpid = process_qrecc_per_dir(commoncrawl_paths, pid, pid2rawpid, fw, zip_file)
            pid, pid2rawpid = process_qrecc_per_dir(wayback_paths, pid, pid2rawpid, fw, zip_file)
            pid, pid2rawpid = process_qrecc_per_dir(wayback_backfill_paths, pid, pid2rawpid, fw, zip_file)

        pstore(pid2rawpid, pid2rawpid_path)

        logger.info("generate QReCC passage collection -> {} ok!".format(output_file))
        logger.info("#totoal passages = {}".format(pid)) # 54573064 (blank 331474)


def gen_qrecc_qrel(train_inputfile, test_inputfile, train_qrel_file, test_qrel_file, pid2rawpid_path):

    pid2rawpid = pload(pid2rawpid_path)
    rawpid2pid = {}
    for pid, rawpid in pid2rawpid.items():
        rawpid2pid[rawpid] = pid

    with open(train_inputfile, "r") as f:
        data = json.load(f)

    with open(train_qrel_file, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}_{}".format("QReCC-Train", line['Conversation_no'], line['Turn_no'])
            for rawpid in line['Truth_passages']:
                f.write("{} {} {} {}".format(sample_id, 0, rawpid2pid[rawpid], 1))
                f.write('\n') 

    logger.info("generate qrecc qrel file -> {} ok!".format(train_qrel_file))

    with open(test_inputfile, "r") as f:
        data = json.load(f)

    with open(test_qrel_file, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}_{}".format("QReCC-Test", line['Conversation_no'], line['Turn_no'])
            for rawpid in line['Truth_passages']:
                f.write("{} {} {} {}".format(sample_id, 0, rawpid2pid[rawpid], 1))
                f.write('\n') 
                
    logger.info("generate qrecc qrel file -> {} ok!".format(test_qrel_file))


def gen_qrecc_train_test_files(train_inputfile,
                               test_inputfile, 
                               train_outputfile, 
                               test_outputfile,
                               pid2rawpid_path):
    
    pid2rawpid = pload(pid2rawpid_path)
    rawpid2pid = {}
    #for pid, rawpid in enumerate(pid2rawpid):
    for pid, rawpid in pid2rawpid.items():
        rawpid2pid[rawpid] = pid

    # train & test raw files
    outputfile2inputfile = {
        train_outputfile: train_inputfile,
        test_outputfile: test_inputfile
    }

    for outputfile in outputfile2inputfile:
        with open(outputfile2inputfile[outputfile], "r") as f:
            data = json.load(f)

        cxt = []
        conv_no, turn_no = None, None
        with open(outputfile, "w") as f:
            for line in tqdm(data):
                record = {}
                sample_title = "QReCC-Train" if outputfile == train_outputfile else "QReCC-Test"
                qid = "{}_{}_{}".format(sample_title, line['Conversation_no'], line['Turn_no'])
                record["qid"] = qid
                record["source"] = line["Conversation_source"]
                record["question"] = line["Question"]
                record["truth_rewrite"] = line["Truth_rewrite"]
                record['conv_no'] = line['Conversation_no']
                record['turn_no'] = line['Turn_no']

                if conv_no is None and turn_no is None:
                    conv_no = line['Conversation_no']
                    turn_no = line['Turn_no']
                elif conv_no != line['Conversation_no']:
                    conv_no = line['Conversation_no']
                    turn_no = line['Turn_no']
                    cxt = []
                else:
                    assert turn_no == line['Turn_no']

                turn_no += 1
                record["context"] = copy.deepcopy(cxt)
                cxt.append(line["Question"])
                cxt.append(line['Truth_answer'] if line['Truth_answer'] != "" else "UNANSWERABLE")

                if len(line['Truth_passages']) == 0:
                    continue

                record["truth_passages"] = [str(rawpid2pid[rawpid]) for rawpid in line['Truth_passages']]
                
                f.write(json.dumps(record))
                f.write('\n')
    
    logger.info("QReCC train test file preprocessing (first stage) ok!")


if __name__ == "__main__":
    input_passage_path = "download/qrecc/passages.zip"
    output_file = "data/qrecc/collection/qrecc_collection.jsonl"
    pid2rawpid_path = "data/qrecc/pid2rawpid.pkl"
    gen_qrecc_passage_collection(input_passage_path, output_file, pid2rawpid_path)
    
    train_inputfile = "download/qrecc/scai-qrecc21-training-turns.json"
    test_inputfile = "download/qrecc/scai-qrecc21-test-turns.json"
    train_outputfile = "data/qrecc/train_v3.json"
    test_outputfile = "data/qrecc/test_v3.json"
    pid2rawpid_path = "data/qrecc/pid2rawpid.pkl"
    gen_qrecc_train_test_files(train_inputfile, test_inputfile, train_outputfile, test_outputfile, pid2rawpid_path)

    train_inputfile = "download/qrecc/scai-qrecc21-training-turns.json"
    test_inputfile = "download/qrecc/scai-qrecc21-test-turns.json"
    train_qrel_file = "data/qrecc/train.trec"
    test_qrel_file = "data/qrecc/test.trec"
    pid2rawpid_path = "data/qrecc/pid2rawpid.pkl"
    gen_qrecc_qrel(train_inputfile, test_inputfile, train_qrel_file, test_qrel_file, pid2rawpid_path)
