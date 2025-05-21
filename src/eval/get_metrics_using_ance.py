import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

import argparse
from models import ANCE
from data_format import Retrieval_Dataset
import os
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import faiss
import time
import pickle
import torch
import numpy as np
from typing import List, Tuple, Any
import pytrec_eval
from transformers import RobertaConfig, RobertaTokenizer

'''
Test process, perform dense retrieval on collection (e.g., MS MARCO):
1. get args
2. establish index with Faiss on GPU for fast dense retrieval
3. load the model, build the test query dataset/dataloader, and get the query embeddings. 
4. iteratively searched on each passage block one by one to got the retrieved scores and passge ids for each query.
5. merge the results on all pasage blocks
6. output the result
'''

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def build_faiss_index(args):
    logger.info("Building index...")
    ngpu = args.n_gpu
    gpu_resources = []
    tempmem = -1

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    cpu_index = faiss.IndexFlatIP(768)
    index = None
    if args.use_gpu:
        logger.info("Use GPUs...")
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        # gpu_vector_resources, gpu_devices_vector
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres, vdev, cpu_index, co)
        index = gpu_index
    else:
        index = cpu_index
    return index


def search_one_by_one_with_faiss(args, passge_embeddings_dir, index, query_embeddings: List[np.ndarray], topN):

    start_time = time.time()
    with open(os.path.join(passge_embeddings_dir, "doc_emb_cat.pkl"), 'rb') as f:
        passage_embedding = pickle.load(f)
    with open(os.path.join(passge_embeddings_dir, "doc_embid_cat.pkl"), 'rb') as f:
        passage_embedding2id = pickle.load(f)
    logger.info('Loading time: ' + str(time.time() - start_time))
    logger.info('passage embedding shape: ' + str(passage_embedding.shape))
    logger.info("query embedding shape: " + str(query_embeddings[0].shape))
    index.add(passage_embedding)

    # ann search
    tb = time.time()
    all_D, all_I = [], []
    for query_emb in query_embeddings:
        D, I = index.search(query_emb.astype('float32'), topN)
        all_D.append(D)
        all_I.append(I)
    elapse = time.time() - tb
    logger.info({
        'time cost': elapse,
        'query num': query_embeddings[0].shape,
        'time cost per query': elapse / query_embeddings[0].shape[0]
    })

    all_candidate_id_matrix = []
    for I in all_I:
        candidate_id_matrix = passage_embedding2id[I] # passage_idx -> passage_id
        all_candidate_id_matrix.append(candidate_id_matrix)
    index.reset()
    logger.info(all_candidate_id_matrix[0].shape)
    return all_D, all_candidate_id_matrix


def get_test_query_embedding(args) -> Tuple[List[np.ndarray], List[List[List[str]]]]:
    set_seed(args)

    config = RobertaConfig.from_pretrained(args.pretrained_encoder_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_encoder_path, do_lower_case=True)
    model = ANCE.from_pretrained(args.pretrained_encoder_path, config=config).to(args.device)

    args.batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)
    logger.info("Buidling test dataset...")

    all_embeddings, all_embedding2id = [], []
    for test_file_path in args.test_file_path:
        test_dataset = Retrieval_Dataset(args.max_concat_length, tokenizer, test_file_path)
        test_loader = DataLoader(
            test_dataset,
            batch_size = args.batch_size, 
            shuffle=False, 
            collate_fn=test_dataset.get_collate_fn(args)
        )

        logger.info("Generating query embeddings for testing...")
        model.zero_grad()
        embeddings = []
        embedding2id = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                model.eval()
                bt_sample_ids = batch["bt_sample_ids"] # question id
                # test type
                if args.test_type == "rewrite":
                    input_ids = batch["bt_rewrite"].to(args.device)
                    input_masks = batch["bt_rewrite_mask"].to(args.device)
                else:
                    raise ValueError("test type:{}, has not been implemented.".format(args.test_type))
                
                query_embs = model(input_ids, input_masks)
                query_embs = query_embs.detach().cpu().numpy()
                embeddings.append(query_embs)
                embedding2id.extend(bt_sample_ids)

        embeddings = np.concatenate(embeddings, axis=0)
        all_embeddings.append(embeddings)
        all_embedding2id.append(embedding2id)

    torch.cuda.empty_cache()
    del model
    return all_embeddings, all_embedding2id


def output_test_res(
    alll_query_embedding2id: List[np.ndarray],
    all_retrieved_scores_mat: List[Any], # score_mat: score matrix, test_query_num * (top_k * block_num)
    all_retrieved_pid_mat: List[Any], # pid_mat: corresponding passage ids
    args
):
    for query_embedding2id, retrieved_scores_mat, retrieved_pid_mat, output_trec_file in zip(
        alll_query_embedding2id, 
        all_retrieved_scores_mat, 
        all_retrieved_pid_mat, 
        args.output_trec_file
    ):
        qids_to_ranked_candidate_passages = {}
        topN = args.top_k

        for query_idx in range(len(retrieved_pid_mat)):
            seen_pid = set()
            query_id = query_embedding2id[query_idx]

            top_ann_pid = retrieved_pid_mat[query_idx].copy()
            top_ann_score = retrieved_scores_mat[query_idx].copy()
            selected_ann_idx = top_ann_pid[:topN]
            selected_ann_score = top_ann_score[:topN].tolist()
            rank = 0

            if query_id in qids_to_ranked_candidate_passages:
                pass
            else:
                tmp = [(0, 0)] * topN
                qids_to_ranked_candidate_passages[query_id] = tmp
            
            for pred_pid, score in zip(selected_ann_idx, selected_ann_score):
                if not pred_pid in seen_pid:
                    qids_to_ranked_candidate_passages[query_id][rank] = (pred_pid, score)
                    rank += 1
                    seen_pid.add(pred_pid)

        # write to file
        logger.info('begin to write the output...')

        os.makedirs(args.qrel_output_dir, exist_ok=True)
        output_trec_file_path = os.path.join(args.qrel_output_dir, output_trec_file)
        with open(output_trec_file_path, "w") as g:
            for qid, passages in qids_to_ranked_candidate_passages.items():
                for i in range(topN):
                    pid, score = passages[i]
                    g.write(str(qid) + " Q0 " + str(pid) + " " + str(i + 1) + " " + str(-i - 1 + 200) + ' ' + str(score) + " ance\n")

        print_trec_res(output_trec_file_path, args.trec_gold_qrel_file_path, args.rel_threshold)


def print_trec_res(run_file, qrel_file, rel_threshold=1):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    for line in qrel_data:
        line = line.split()
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
        line = line.split()
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
        # "MAP": round(np.average(map_list), 3),
        "MRR": round(np.average(mrr_list), 3),
        "NDCG@3": round(np.average(ndcg_3_list), 3), 
        "Recall@10": round(np.average(recall_10_list), 3),
        "Recall@100": round(np.average(recall_100_list), 3),
        "Recall@5": round(np.average(recall_5_list), 3),
        "Recall@20": round(np.average(recall_20_list), 3),
    }
    
    logger.info("---------------------Evaluation results:---------------------")    
    logger.info(res)


def gen_metric_score_and_save(args, index, query_embeddings: List[np.ndarray], query_embedding2id: List[List[List[str]]]):
    # score_mat: score matrix, test_query_num * (top_n * block_num)
    # pid_mat: corresponding passage ids
    retrieved_scores_mat, retrieved_pid_mat = \
        search_one_by_one_with_faiss(
            args,
            args.passage_embeddings_dir_path, 
            index,
            query_embeddings, 
            args.top_k
        ) 

    output_test_res(
        query_embedding2id,
        retrieved_scores_mat,
        retrieved_pid_mat,
        args
    )


def main():
    args = get_args()
    set_seed(args) 
    index = build_faiss_index(args)
    query_embeddings, query_embedding2id = get_test_query_embedding(args)
    gen_metric_score_and_save(args, index, query_embeddings, query_embedding2id)
    logger.info("Test finish!")
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file_path", type=str, default="data/topiocqa/rewrite/DS-R1-Distill-Qwen-7B.jsonl", nargs='+')
    parser.add_argument("--passage_embeddings_dir_path", type=str, default="embedding/ance_topiocqa/numpy<2.0")
    parser.add_argument("--pretrained_encoder_path", type=str, default="./ckpt/ance/ad-hoc-ance-msmarco")
    parser.add_argument("--qrel_output_dir", type=str, default="data/topiocqa/qrel")
    parser.add_argument("--output_trec_file", type=str, default="DS-R1-Distill-Qwen-7B_fast.trec", nargs='+')
    parser.add_argument("--trec_gold_qrel_file_path", type=str, default="data/topiocqa/dev.trec")
    parser.add_argument("--test_type", type=str, default="rewrite")
    # parser.add_argument("--dataset", type=str, default="topiocqa")
    parser.add_argument("--is_train", type=bool, default=False)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--n_gpu", type=int, default=3)
    parser.add_argument("--rel_threshold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_gpu_test_batch_size", type=int, default=32)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_doc_length", type=int, default=384)
    parser.add_argument("--max_response_length", type=int, default=64)
    parser.add_argument("--max_concat_length", type=int, default=512)
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    args.device = device

    assert len(args.test_file_path) == len(args.output_trec_file)

    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    return args

if __name__ == '__main__':
    main()
    # if you already have the trec file, run following code directly to get the metrics.
    # print_trec_res("", "", rel_threshold=1)
    