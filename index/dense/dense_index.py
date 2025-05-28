import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import time
import argparse
import numpy as np
from os.path import join as oj
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from utils import set_seed, check_dir_exist_or_build, json_dumps_arguments, pstore
from models import load_model
from libs import StreamIndexDataset, CollateClass

def dense_indexing(args):
    tokenizer, model = load_model(args.model_type, "doc", args.pretrained_doc_encoder_path)
    model = torch.nn.DataParallel(model)
    model.to(args.device)
    
    indexing_batch_size = args.per_gpu_index_batch_size
    indexing_dataset = StreamIndexDataset(args.collection_path, args.dataset)
    if args.model_type == "TCT-ColBERT":
        prefix = "[ D ] "   # note that [CLS] will be added by tokenizer with the "add_special_token" param
    else:
        prefix = ""
    collate_func = CollateClass(args, tokenizer, prefix=prefix)
    index_dataloader = DataLoader(
        indexing_dataset, 
        batch_size=indexing_batch_size, 
        collate_fn=collate_func.collate_fn
    )

    doc_ids = []
    doc_embeddings = []
    cur_block_id = 0
    num_doc_embs = 0
    num_per_block_docs = 5000000 # 3844000 is ~6.9GB
    with torch.no_grad():
        model.eval()
        for batch in tqdm(index_dataloader, desc="Dense Indexing", position=0, leave=True, total=len(index_dataloader)):
      
            inputs = {k: v.to(args.device) for k, v in batch.items() if k not in {"id"}}
            batch_doc_embs = model(**inputs)
            batch_doc_embs = batch_doc_embs.detach().cpu().numpy()
            doc_embeddings.append(batch_doc_embs)

            for doc_id in batch["id"]:
                doc_ids.append(doc_id)
        
            if len(doc_ids) >= num_per_block_docs:
                doc_embeddings = np.concatenate(doc_embeddings, axis=0)
                doc_ids = np.array(doc_ids)
                emb_output_path = oj(args.output_index_dir_path, "doc_emb_block.{}.pb".format(cur_block_id))
                embid_output_path = oj(args.output_index_dir_path, "doc_embid_block.{}.pb".format(cur_block_id))
                pstore(doc_embeddings, emb_output_path, high_protocol=True)
                pstore(doc_ids, embid_output_path, high_protocol=True)
                
                num_doc_embs += len(doc_ids)
                doc_ids = []
                doc_embeddings = []
                cur_block_id += 1
    
    if len(doc_ids) > 0:
        doc_embeddings = np.concatenate(doc_embeddings, axis=0)
        doc_ids = np.array(doc_ids)
        emb_output_path = oj(args.output_index_dir_path, "doc_emb_block.{}.pb".format(cur_block_id))
        embid_output_path = oj(args.output_index_dir_path, "doc_embid_block.{}.pb".format(cur_block_id))
        pstore(doc_embeddings, emb_output_path, high_protocol=True)
        pstore(doc_ids, embid_output_path, high_protocol=True)    

        num_doc_embs += len(doc_ids)
        doc_ids = []
        doc_embeddings = []
        cur_block_id += 1
    
    print("Totally {} docs in {} blocks are stored.".format(num_doc_embs, cur_block_id))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["cast19", "cast20", "qrecc", "topiocqa"])
    parser.add_argument("--model_type", type=str, default='ance')
    parser.add_argument("--collection_path", type=str, default="cast20/collection/collection.tsv")
    parser.add_argument("--pretrained_doc_encoder_path", type=str, default="3ricL/ad-hoc-ance-msmarco")
    parser.add_argument("--output_index_dir_path", type=str, default="embedding/ance_topiocqa")
    parser.add_argument("--force_emptying_dir", action="store_true", default=True) # type: <class 'bool'>
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_data_percent", type=float, default=1.0, help="Percent of samples to use. Faciliating the debugging.")
    parser.add_argument("--per_gpu_index_batch_size", type=int, default=1024)
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length, consistent with \"Dialog inpainter\".")
    args = parser.parse_args()
    # pytorch parallel gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.start_running_time = time.asctime(time.localtime(time.time()))
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)

    check_dir_exist_or_build([args.output_index_dir_path], force_emptying=args.force_emptying_dir)
    json_dumps_arguments(oj(args.output_index_dir_path, "parameters.txt"), args)
        
    return args


if __name__ == "__main__":
    args = get_args()
    set_seed(args)
    dense_indexing(args)
