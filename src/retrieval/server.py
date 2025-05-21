# This script is adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

import torch
from torch import nn
from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
import uvicorn
from fastapi import FastAPI
import argparse
from typing import List, Optional
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import faiss
import pickle
import time
import json
from pyserini.search.lucene import LuceneSearcher


class ANCE(RobertaForSequenceClassification):
    def __init__(self, config):
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        self.use_mean = False
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        outputs1 = outputs1.last_hidden_state
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def doc_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)
    
    def masked_mean_or_first(self, emb_all, mask):
        if self.use_mean:
            return self.masked_mean(emb_all, mask)
        else:
            return emb_all[:, 0]
    
    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    
    def forward(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)
      
########################### utils start #############################
def load_model(model_path: str):
    config = RobertaConfig.from_pretrained(
        model_path,
        finetuning_task="MSMarco",
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        model_path,
        do_lower_case=True
    )
    model = ANCE.from_pretrained(model_path, config=config).to('cuda')
    return tokenizer, model

def build_faiss_index(index_path: str, n_gpu: int, faiss_gpu: bool=True):
    gpu_resources = []
    tempmem = -1
    for i in range(n_gpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    cpu_index = faiss.IndexFlatIP(768)  
    index = None
    if faiss_gpu:
        logger.info("Using GPU for FAISS")
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, n_gpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres, vdev, cpu_index, co)
        index = gpu_index
    else:
        index = cpu_index

    logger.info("Loading embedding")

    start = time.time()
    with open(index_path, 'rb') as f:
        passage_embedding = pickle.load(f)
    logger.info("Embedding loaded in %.2f seconds", time.time() - start)
    index.add(passage_embedding)
    return index

def load_docid(corpus_docid_path: str):
    with open(corpus_docid_path, 'rb') as f:
        passage_embedding2id = pickle.load(f)
    return passage_embedding2id

def load_gold(trec_path: str):
    qid2goldid = {}
    with open(trec_path, 'r') as f:
        for line in f:
            line = line.split()
            qid = line[0]
            goldid = line[2]
            if qid in qid2goldid:
                qid2goldid[qid].append(goldid)
            else:
                qid2goldid[qid] = [goldid]
    return qid2goldid

class QueryRequest(BaseModel):
    queries: List[str]
    query_ids: List[str]
    topk: Optional[int] = None
    return_scores: bool = False
    max_length: Optional[int] = 1024 # for bm25
########################### utils end #############################

class Encoder:
    def __init__(self, model_path, max_length):
        self.max_length = max_length
        self.tokenizer, self.model = load_model(model_path=model_path)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str]):
        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        ).to('cuda')
        query_embs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        query_embs = query_embs.detach().cpu().numpy()
        return query_embs


class BaseRetriever:
    def __init__(self):
        raise NotImplementedError
    
    def _batch_search(self, query_list: List[str], query_ids: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def batch_search(self, query_list: List[str], query_ids: List[str], num: int=100, return_score: bool=True, max_length: int=1024):
        return self._batch_search(query_list, query_ids, num, return_score, max_length)


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        self.qid2goldid = load_gold(config.trec_path)
        self.searcher = LuceneSearcher(config.index_path)
        self.searcher.set_bm25(config.bm25_k1, config.bm25_b)

    def _get_gold_rank(self, candidate_ids: List[List[str]], query_ids_batch: List[str]):
        gold_ranks = []
        for i, query_id in enumerate(query_ids_batch):
            gold_id = self.qid2goldid[query_id] # type: list
            try:
                gold_rank = []
                for id in gold_id:
                    if id in candidate_ids[i]:
                        gold_rank.append(candidate_ids[i].index(id) + 1)
                    else:
                        gold_rank.append(200)
                gold_rank = min(gold_rank)
                if gold_rank == 200:
                    gold_rank = -1
                # logger.info(candidate_ids[i])
            except ValueError:
                gold_rank = -1  # -1 means not found
            gold_ranks.append(gold_rank)
        return gold_ranks

    def _batch_search(self, query_list: List[str], query_ids: List[str], num: int=100, return_score: bool=True, max_length: int=1024):
        if isinstance(query_list, str):
            query_list = [query_list]

        cut_queries = []
        for query in query_list:
            query = query.split(" ")[:max_length]
            query = " ".join(query)
            cut_queries.append(query)
        # hits = self.searcher.batch_search(cut_queries, query_ids, k=num, threads=30)

        hits = {}
        batch_size = 100
        for i in range(0, len(cut_queries), batch_size):
            batch_cut_queries = cut_queries[i:i + batch_size]
            batch_qid_list = query_ids[i:i + batch_size]
            batch_hits = self.searcher.batch_search(batch_cut_queries, batch_qid_list, k=num, threads=20)
            hits.update(batch_hits)

        scores = []
        candidate_ids = []
        for qid in query_ids:
            item_scores = []
            item_candidate_ids = []
            if qid not in hits:
                candidate_ids.append([])
                scores.append([])
            for item in hits[qid]:
                item_scores.append(item.score)
                item_candidate_ids.append(json.loads(self.searcher.doc(item.docid).raw())['id'])
            candidate_ids.append(item_candidate_ids)
            scores.append(item_scores)
        gold_ranks = self._get_gold_rank(candidate_ids, query_ids)
            
        if return_score:
            return gold_ranks, scores
        else:
            return gold_ranks


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        self.index = build_faiss_index(config.index_path, config.n_gpu, config.faiss_gpu)
        self.encoder = Encoder(
            model_path=config.model_name_or_path,
            max_length=config.max_length,
        )
        self.batch_size = config.retrieval_batch_size
        self.embed2id = load_docid(config.corpus_docid_path)
        self.qid2goldid = load_gold(config.trec_path)

    def _get_gold_rank(self, candidate_ids: List[List[str]], query_ids_batch: List[str]):
        gold_ranks = []
        for i, query_id in enumerate(query_ids_batch):
            gold_id = self.qid2goldid[query_id] # type: list
            try:
                gold_rank = []
                for id in gold_id:
                    if id in candidate_ids[i]:
                        gold_rank.append(candidate_ids[i].index(id) + 1)
                    else:
                        gold_rank.append(200)
                gold_rank = min(gold_rank)
                if gold_rank == 200:
                    gold_rank = -1
                # logger.info(candidate_ids[i])
            except ValueError:
                gold_rank = -1  # -1 means not found
            gold_ranks.append(gold_rank)
        return gold_ranks

    def _batch_search(self, query_list: List[str], query_ids: List[str], num: int=100, return_score: bool=True, max_length: int=1024):
        if isinstance(query_list, str):
            query_list = [query_list]
        
        scores = []
        gold_ranks = []
        for start_idx in range(0, len(query_list), self.batch_size):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            query_ids_batch = query_ids[start_idx:start_idx + self.batch_size]
            
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            candidate_ids = self.embed2id[batch_idxs]
            batch_ranks = self._get_gold_rank(candidate_ids.tolist(), query_ids_batch)

            scores.extend(batch_scores)
            gold_ranks.extend(batch_ranks)
            torch.cuda.empty_cache()
            
        if return_score:
            return gold_ranks, scores
        else:
            return gold_ranks

    
def get_retriever(config):
    if config.retriever_name == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)

app = FastAPI()

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest) -> JSONResponse:
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    # Perform batch retrieval
    results = retriever.batch_search(
        query_list=request.queries,
        query_ids=request.query_ids,
        num=request.topk,
        return_score=request.return_scores,
        max_length=request.max_length
    )
    
    return JSONResponse({"result": results})
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch the local FAISS or Pyserini retriever.")
    parser.add_argument("--index_path", type=str, required=True, help="Corpus indexing file.")
    parser.add_argument('--trec_path', type=str, required=True, help='A file that includes ids of gold docs of each query.')
    parser.add_argument('--corpus_docid_path', type=str, help='A file that includes docid of each doc.')
    parser.add_argument("--retriever_name", type=str, default='ance', help="Name of the retriever model.")
    parser.add_argument("--model_name_or_path", type=str, default="3ricL/ad-hoc-ance-msmarco", help="Path of the retriever model.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for FAISS.')
    parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs to use.')
    parser.add_argument('--max_length', type=int, default=512, help='Max length of the query.')
    parser.add_argument('--retrieval_batch_size', type=int, default=64, help='Batch size for retrieval.')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--host', type=str, default='::')

    bm25_parser = parser.add_argument_group('bm25')
    bm25_parser.add_argument("--bm25_k1", type=float, default=0.9)   # 0.82 for qrecc, 0.9 for topiocqa
    bm25_parser.add_argument("--bm25_b", type=float, default=0.4)    # 0.68 for qrecc, 0.4 for topiocqa
    args = parser.parse_args()

    retriever = get_retriever(args)

    # Launch the server. By default, it listens on http://127.0.0.1:8000
    # host = "::" if args.retriever_name == 'ance' else "0.0.0.0"
    # host = "::"
    # print(f'host: {host}')
    uvicorn.run(app, host=args.host, port=args.port)
