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

from verl import DataProto
import torch


class RewriteR1RewardManager:
    """The reward manager.
    """

    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        compute_score,
        retriever_url,
        topk,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score
        self.retriever_url = retriever_url
        self.topk = topk

    def __call__(self, data: DataProto, step: int=None, total_steps: int=None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # batch compute
        prompt_strs = []
        response_strs = []
        query_ids = []
        valid_response_lengths = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True) 
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            
            prompt_strs.append(prompt_str)
            response_strs.append(response_str)
            query_ids.append(extra_info['query_id'])
            valid_response_lengths.append(valid_response_length)

        scores = self.compute_score(
            response_strs=response_strs, 
            query_ids=query_ids, 
            retriever_url=self.retriever_url,
            topk=self.topk,
            step=step,
            total_steps=total_steps,
        )
        for i, valid_response_length in enumerate(valid_response_lengths):
            reward_tensor[i, valid_response_length - 1] = scores[i]

        print("[prompt]", prompt_strs[-1])
        print("[response]", response_strs[-1])
        print("[score]", scores[-1])

        return reward_tensor
