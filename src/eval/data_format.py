import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import Dataset
import json


def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids

    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length
  
    return input_ids, attention_mask


class Retrieval_Dataset(Dataset):
    def __init__(self, max_concat_length, tokenizer, filename):
        self.examples = []
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)

        for i in range(n):
            record = json.loads(data[i])
            sample_id = record['qid']
            rewite = record['rewrite']
            # rewite = record['truth_rewrite']
            rewrite_encoded = tokenizer.encode(rewite, add_special_tokens=True)
            rewrite_padded, rewrite_mask = padding_seq_to_same_length(rewrite_encoded, max_pad_length=max_concat_length)
            self.examples.append([sample_id, rewrite_padded, rewrite_mask])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_ids": [],
                "bt_rewrite":[],
                "bt_rewrite_mask":[],
            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_rewrite"].append(example[1])
                collated_dict["bt_rewrite_mask"].append(example[2])
                
            for key in collated_dict:
                if key != 'bt_sample_ids':
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn
