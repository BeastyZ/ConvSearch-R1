import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from os.path import join as oj
import os

import json
import argparse
import numpy as np
from pprint import pprint
import jsonlines

from pyserini.search.lucene import LuceneSearcher
import pytrec_eval

def run_bm25(args):

    query_lists, qid_lists = [], []
    for query_file_path in args.query_file:
        query_list, qid_list = [], []
        with jsonlines.open(query_file_path, "r") as f:
            for ln in f:
                rewrite = ln['rewrite']
                # rewrite = ln['truth_rewrite']
                rewrite = rewrite.split(" ")[:args.max_seq_length]
                rewrite = " ".join(rewrite)
                query_list.append(rewrite)
                qid_list.append(ln['qid'])
        query_lists.append(query_list)
        qid_lists.append(qid_list)
            
    # pyserini search
    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)

    all_hits = []
    batch_size = 100
    for query_list, qid_list in zip(query_lists, qid_lists):
        hits = {}
        for i in range(0, len(query_list), batch_size):
            batch_query_list = query_list[i:i + batch_size]
            batch_qid_list = qid_list[i:i + batch_size]
            batch_hits = searcher.batch_search(batch_query_list, batch_qid_list, k=args.top_n, threads=30)
            hits.update(batch_hits)
        all_hits.append(hits)

    for file_name, qid_list, hits in zip(args.output_file_name, qid_lists, all_hits):
        save_path = oj(args.retrieval_output_path, file_name)
        with open(save_path, "w") as f:
            for qid in qid_list:
                if qid not in hits:
                    print("{} not in hits".format(qid))
                    continue
                for i, item in enumerate(hits[qid]):
                    f.write(
                        "{} {} {} {} {} {}".format(
                            qid,
                            "Q0",
                            # item.docid,
                            json.loads(searcher.doc(item.docid).raw())['id'],
                            i + 1,
                            -i - 1 + 200,
                            item.score,
                            "bm25"
                        )
                    )
                    f.write('\n')

        if not args.not_perform_evaluation:
            print_res(save_path, args.gold_qrel_file_path, args.rel_threshold)
    

def print_res(run_file, qrel_file, rel_threshold):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    for line in qrel_data:
        line = line.strip().split()
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(line[4])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    res = {
            #"MAP": np.average(map_list),
            "MRR": round(np.average(mrr_list), 3),
            "NDCG@3": round(np.average(ndcg_3_list), 3), 
            "Recall@10": round(np.average(recall_10_list), 3),
            "Recall@100": round(np.average(recall_100_list), 3),
            "Recall@5": round(np.average(recall_5_list), 3),
            "Recall@20": round(np.average(recall_20_list), 3),
        }
    
    logger.info("---------------------Evaluation results:---------------------")    
    logger.info(res)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir_path", type = str, default="/media/nvme/fengran/index/bm25_topiocqa")
    parser.add_argument("--query_file", type = str, default="/media/nvme/fengran/output/combine_topiocqa/mistral_TopiOCQA_test_AD+FT.json", nargs='+')
    parser.add_argument("--gold_qrel_file_path", type=str, default="/media/nvme/fengran/TopiOCQA/topiocqa_qrel.tsv")
    parser.add_argument("--not_perform_evaluation", action="store_true", default=False)
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--bm25_k1", type=float, default=0.9)   # 0.82 for qrecc, 0.9 for topiocqa
    parser.add_argument("--bm25_b", type=float, default=0.4)    # 0.68 for qrecc, 0.4 for topiocqa
    parser.add_argument("--rel_threshold", type=int, default=1)
    parser.add_argument("--split_num_chunk", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--retrieval_output_path", type=str, default="data/topiocqa/qrel/bm25")
    parser.add_argument("--output_file_name", type=str, default="mistral_TopiOCQA_test_AD+FT.trec", nargs='+')
    args = parser.parse_args()

    logger.info("---------------------The arguments are:---------------------")
    pprint(args)
    assert len(args.query_file) == len(args.output_file_name)
    return args

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.retrieval_output_path, exist_ok=True)
    run_bm25(args)
    # if you already have the trec file, run following code directly to get the metrics.
    # print_res(
    #     "", 
    #     "", 
    #     rel_threshold=1
    # )
