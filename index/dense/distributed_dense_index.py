import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import jsonlines

from utils import set_seed, pstore
from models import load_model
from libs import CollateClass


def distributed_index_dataset_generator(collection_path, num_doc_per_block):
    with jsonlines.open(collection_path, "r") as f:
        docs = []
        for line in f:
            docs.append([line['id'], line['contents']])
            if len(docs) == num_doc_per_block:
                yield docs
                docs = []
        yield docs


def dense_indexing(args, local_rank, world_size):

    dist.init_process_group(backend='nccl', init_method='env://', rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    tokenizer, model = load_model(args.model_type, "doc", args.pretrained_doc_encoder_path)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    indexing_dataset_generator = distributed_index_dataset_generator(args.collection_path, args.num_docs_per_block)
    collate_func = CollateClass(args, tokenizer, prefix='')

    for cur_block_id, raw_docs in enumerate(indexing_dataset_generator):
        distributed_sampler = DistributedSampler(raw_docs, shuffle=False)
        dataloader =  DataLoader(
            raw_docs, 
            sampler=distributed_sampler,
            batch_size=args.per_gpu_index_batch_size, 
            collate_fn=collate_func.collate_fn,
            shuffle=False
        )
        
        doc_ids = []
        doc_embeddings = []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(dataloader, desc="Distributed Dense Indexing", position=0, leave=True):
                inputs = {k: v.to(local_rank) for k, v in batch.items() if k not in {"id"}}
                batch_doc_embs = model(**inputs)
                batch_doc_embs = batch_doc_embs.detach().cpu().numpy()
                doc_embeddings.append(batch_doc_embs)
                for doc_id in batch["id"]:
                    doc_ids.append(doc_id)

        doc_embeddings = np.concatenate(doc_embeddings, axis=0)
        doc_ids = np.array(doc_ids)
        emb_output_path = os.path.join(args.output_index_dir_path, "doc_emb_block.rank_{}.{}.pkl".format(dist.get_rank(), cur_block_id))
        embid_output_path = os.path.join(args.output_index_dir_path, "doc_embid_block.rank_{}.{}.pkl".format(dist.get_rank(), cur_block_id))
        pstore(doc_embeddings, emb_output_path, high_protocol=True)
        pstore(doc_ids, embid_output_path, high_protocol=True)

    dist.barrier()
    dist.destroy_process_group()
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="ance")
    parser.add_argument("--collection_path", type=str, default="cast20/collection/collection.tsv")
    parser.add_argument("--pretrained_doc_encoder_path", type=str, default="checkpoints/ad-hoc-ance-msmarco")
    parser.add_argument("--output_index_dir_path", type=str, default="index/ance_cast1920")
    parser.add_argument("--force_emptying_dir", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--per_gpu_index_batch_size", type=int, default=250)
    parser.add_argument("--num_docs_per_block", type=int, default=5000000)
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length, consistent with \"Dialog inpainter\".")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_index_dir_path, exist_ok=True)
    
    world_size = int(os.environ.get("WORLD_SIZE", 1)) # Automatic setting
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) # Automatic setting
    print(f"Using {world_size} GPUs for encoding... Local Rank: {local_rank}")
    
    dense_indexing(args, local_rank, world_size)
