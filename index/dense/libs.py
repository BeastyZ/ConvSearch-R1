import os
import h5py
import json
import numba
import array
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch
from torch.utils.data import IterableDataset
import jsonlines

from utils import tensor_to_list

class IndexDictOfArray:
    def __init__(self, index_path=None, force_new=False, filename="array_index.h5py", dim_voc=None):
        # index_path = None # for debug
        if index_path is not None:
            self.index_path = index_path
            if not os.path.exists(index_path):
                os.makedirs(index_path)
            self.filename = os.path.join(self.index_path, filename)
            if os.path.exists(self.filename) and not force_new:
                print("index already exists, loading...")
    
                self.file = h5py.File(self.filename, "r")
                if dim_voc is not None:
                    dim = dim_voc
                else:
                    dim = self.file["dim"][()]
                self.index_doc_id = dict()
                self.index_doc_value = dict()
                for key in tqdm(range(dim)):
                    try:
                        self.index_doc_id[key] = np.array(self.file["index_doc_id_{}".format(key)],
                                                          dtype=np.int32)
                        # ideally we would not convert to np.array() but we cannot give pool an object with hdf5
                        self.index_doc_value[key] = np.array(self.file["index_doc_value_{}".format(key)],
                                                             dtype=np.float32)
                    except:
                        self.index_doc_id[key] = np.array([], dtype=np.int32)
                        self.index_doc_value[key] = np.array([], dtype=np.float32)
                self.file.close()
                del self.file
                print("done loading index...")
                doc_ids = pickle.load(open(os.path.join(self.index_path, "doc_ids.pkl"), "rb"))
                self.n = len(doc_ids)
            else:
                self.n = 0
                print("initializing new index...")
                self.index_doc_id = defaultdict(lambda: array.array("I"))
                self.index_doc_value = defaultdict(lambda: array.array("f"))
        else:
            self.n = 0
            print("initializing new index...")
            self.index_doc_id = defaultdict(lambda: array.array("I"))
            self.index_doc_value = defaultdict(lambda: array.array("f"))

    def add_batch_document(self, row, col, data, n_docs=-1):
        """add a batch of documents to the index
        """
        if n_docs < 0:
            self.n += len(set(row))
        else:
            self.n += n_docs
        for doc_id, dim_id, value in zip(row, col, data):
            self.index_doc_id[dim_id].append(doc_id)
            self.index_doc_value[dim_id].append(value)

    def __len__(self):
        return len(self.index_doc_id)

    def nb_docs(self):
        return self.n

    def save(self, dim=None):
        print("converting to numpy")
        for key in tqdm(list(self.index_doc_id.keys())):
            self.index_doc_id[key] = np.array(self.index_doc_id[key], dtype=np.int32)
            self.index_doc_value[key] = np.array(self.index_doc_value[key], dtype=np.float32)
        print("save to disk")
        with h5py.File(self.filename, "w") as f:
            if dim:
                f.create_dataset("dim", data=int(dim))
            else:
                f.create_dataset("dim", data=len(self.index_doc_id.keys()))
            for key in tqdm(self.index_doc_id.keys()):
                f.create_dataset("index_doc_id_{}".format(key), data=self.index_doc_id[key])
                f.create_dataset("index_doc_value_{}".format(key), data=self.index_doc_value[key])
            f.close()
        print("saving index distribution...")  # => size of each posting list in a dict
        index_dist = {}
        for k, v in self.index_doc_id.items():
            index_dist[int(k)] = len(v)
        json.dump(index_dist, open(os.path.join(self.index_path, "index_dist.json"), "w"))



class StreamIndexDataset(IterableDataset):
    def __init__(self, collection_path, dataset):
        super().__init__()
        self.collection_path = collection_path
        self.dataset = dataset
        self.collection = []
        if self.dataset == 'topiocqa' or self.dataset == 'qrecc':
            with jsonlines.open(collection_path) as fin:
                for ln in fin:
                    self.collection.append([ln['id'], ln['contents']])
        else:
            raise NotImplementedError
        
    def __len__(self):
        return len(self.collection)

    def __iter__(self):
        """
        Return:
            (id, text with title)
        """
        for ln in self.collection:
            yield ln


class CollateClass:
    def __init__(self, args, tokenizer, prefix=""):
        self.args = args
        self.doc_prefix = prefix
        self.tokenizer = tokenizer
        self.max_doc_length = args.max_doc_length

    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, docs = zip(*batch)
        docs = list(docs)
        if len(self.doc_prefix) > 0:
            for i in range(len(docs)):
                docs[i] = self.doc_prefix + docs[i]
        
        tokenized_docs = self.tokenizer(docs,
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_doc_length,
                                        return_attention_mask=True)
        return {
            **{k: torch.tensor(v) for k, v in tokenized_docs.items()},
            "id": id_
        }


class SparseRetrieval:
    """retrieval from SparseIndexing
    """

    @staticmethod
    def select_topk(filtered_indexes, scores, k):
        if len(filtered_indexes) > k:
            sorted_ = np.argpartition(scores, k)[:k]
            filtered_indexes, scores = filtered_indexes[sorted_], -scores[sorted_]
        else:
            scores = -scores
        return filtered_indexes, scores

    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def numba_score_float(inverted_index_ids: numba.typed.Dict,
                          inverted_index_floats: numba.typed.Dict,
                          indexes_to_retrieve: np.ndarray,
                          query_values: np.ndarray,
                          threshold: float,
                          size_collection: int):
        scores = np.zeros(size_collection, dtype=np.float32)  # initialize array with size = size of collection
        n = len(indexes_to_retrieve)
        for _idx in range(n):
            local_idx = indexes_to_retrieve[_idx]  # which posting list to search
            query_float = query_values[_idx]  # what is the value of the query for this posting list
            retrieved_indexes = inverted_index_ids[local_idx]  # get indexes from posting list
            retrieved_floats = inverted_index_floats[local_idx]  # get values from posting list
            for j in numba.prange(len(retrieved_indexes)):
                scores[retrieved_indexes[j]] += query_float * retrieved_floats[j]
        filtered_indexes = np.argwhere(scores > threshold)[:, 0]  # ideally we should have a threshold to filter
        # unused documents => this should be tuned, currently it is set to 0
        return filtered_indexes, -scores[filtered_indexes]

    def __init__(self, index_dir_path, retrieval_output_path, dim_voc, top_k):
        self.sparse_index = IndexDictOfArray(index_dir_path, dim_voc=dim_voc)
        self.doc_ids = pickle.load(open(os.path.join(index_dir_path, "doc_ids.pkl"), "rb"))
        self.top_k = top_k
        self.retrieval_output_path = retrieval_output_path

        # convert to numba
        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()
        for key, value in self.sparse_index.index_doc_id.items():
            self.numba_index_doc_ids[key] = value
        for key, value in self.sparse_index.index_doc_value.items():
            self.numba_index_doc_values[key] = value
        
    
    def retrieve(self, qid2emb):
        res = defaultdict(dict)
        for qid in tqdm(qid2emb):
            query_emb = qid2emb[qid]
            query_emb = query_emb.view(1, -1)
            row, col = torch.nonzero(query_emb, as_tuple=True)
            values = query_emb[tensor_to_list(row), tensor_to_list(col)]
            threshold = 0
            filtered_indexes, scores = self.numba_score_float(self.numba_index_doc_ids,
                                                                self.numba_index_doc_values,
                                                                col.cpu().numpy(),
                                                                values.cpu().numpy().astype(np.float32),
                                                                threshold=threshold,
                                                                size_collection=self.sparse_index.nb_docs())
            # threshold set to 0 by default, could be better
            filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=self.top_k)
            for id_, sc in zip(filtered_indexes, scores):
                res[str(qid)][str(self.doc_ids[id_])] = float(sc)
            

        with open(os.path.join(self.retrieval_output_path, "run.json"), "w") as f:
            json.dump(res, f)
        print("Write the retrieval result to {} successfully.".format(self.retrieval_output_path))

        return res
        

PyTorch_over_1_6 = float((torch.__version__.split('.')[1])) >= 6 and float((torch.__version__.split('.')[0])) >= 1

# replace this with  contextlib.nullcontext if python >3.7
# see https://stackoverflow.com/a/45187287
class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


class MixedPrecisionManager:
    def __init__(self, activated, use_cuda):
        assert (not activated) or PyTorch_over_1_6, "Cannot use AMP for PyTorch version < 1.6"

        print("Using FP16:", activated)
        self.activated = activated and use_cuda
        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        return torch.cuda.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer):
        if self.activated:
            self.scaler.unscale_(optimizer)
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()
