# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the topiocqa dataset to parquet format
"""

import os
import datasets
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/topiocqa/sft')
    parser.add_argument('--file_path', default='data/topiocqa/sft/llama3.2-3b-it_self.jsonl')
    args = parser.parse_args()

    train_dataset = datasets.load_dataset("json", data_files=args.file_path)['train']

    INSTRUCTION = '''Given a query and its context, you must first think about the reasoning process in the mind to decontextualize the query by resolving \
coreference and omission issues. Then, provide the user with a rewrite that retains its original meaning and is as informative as possible to help \
search engines retrieve relevant documents effectively. The reasoning process and rewrite should be enclosed within <think> </think> and <rewrite> </rewrite> tags, respectively, i.e., \
<think> reasoning process here </think>\n<rewrite> rewrite here </rewrite>.

### Context Begin ###
{context}
### Context End ###

Query: {query}
Rewrite:'''

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            ctx = []
            for i in range(0, len(example['context']), 2):
                ctx.append(f'Q{i // 2 + 1}: {example["context"][i]}')
                ctx.append(f'A{i // 2 + 1}: {example["context"][i + 1]}')
            ctx = '\n'.join(ctx)
            prompt = INSTRUCTION.format(context=ctx, query=example['question'])
            data = {
                "extra_info": {
                     "prompt": prompt,
                    'answer': f'<think>{example["reasoning_content"]}</think>\n<rewrite>{example["rewrite"]}</rewrite>'
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = train_dataset.select(range(1000))
    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train_llama3.2-3b-it_self.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test_llama3.2-3b-it_self.parquet'))
