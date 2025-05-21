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

import re
import requests
from typing import List
import numpy as np
import math


def format_reward(response_strs: List[str], format_score=0.1) -> List[float]:
    # adapted from open-r1
    pattern = r"^<think>.*?</think>\n<rewrite>.*?</rewrite>$"
    matches = [re.match(pattern, response_str, re.DOTALL | re.MULTILINE) for response_str in response_strs]
    return [0.0 if match else -0.1 for match in matches]


def cal_reward(rank: int) -> float:
    if 1 <= rank <= 10:
        return (-1 / 9) * rank + (19 / 9)
    elif 10 < rank <= 100:
        return (-1 / 90) * rank + (10 / 9)
    else:
        return 0.0


def rewrite_reward(response_strs: List[str], query_ids: List[str], retriever_url: str, topk: int=100) -> List[float]:
    pattern = r"<rewrite>(.*?)</rewrite>"
    matches = [re.search(pattern, response_str, re.DOTALL) for response_str in response_strs]
    rewards = [0.0 for _ in range(len(matches))]
    valid_rewrites = [match.group(1) for match in matches if match]
    valid_query_ids = [query_ids[i] for i, match in enumerate(matches) if match]
    payload = {
        "queries": valid_rewrites,
        "query_ids": valid_query_ids,
        "topk": topk,
        "return_scores": False
    }
    response = requests.post(retriever_url, json=payload)
    response.raise_for_status()
    rewrite_ranks: List[int] = response.json()['result']
    rank_cnt = 0
    for i, match in enumerate(matches):
        if match:
            rewards[i] = cal_reward(rewrite_ranks[rank_cnt])
            rank_cnt += 1
    return rewards


def language_reward(response_strs: List[str]) -> List[float]:
    pattern = r"<think>(.*?)</think>"
    rewards = []
    for response_str in response_strs:
        match = re.search(pattern, response_str, re.DOTALL)
        if match:
            think_str = match.group(1)
        else:
            think_str = ''
            
        total_chars = len(think_str)
        english_pattern = re.compile(r'[\w\s,.!?;:\'"()@#\$%^&*_+=<>/\\|~`\[\]{}]', re.UNICODE)
        target_count = len(english_pattern.findall(think_str))
        rewards.append(1.0 if total_chars > 0 and (target_count / total_chars > 0.99) else 0.0)

    return rewards


def cur_cal_reward(rank: int, step: int, total_steps: int) -> float:
    if step <= total_steps // 4:
        if 1 <= rank <= 100:
            return 1.0 - (rank - 1) / 99.0
        else:
            return -0.2
    else:
        if 1 <= rank <= 10:
            return 1.0 - (rank - 1) / 99.0
        elif 10 < rank <= 100:
            return 0.0
        else:
            return -0.2
    

def cur_rewrite_reward(response_strs: List[str], query_ids: List[str], retriever_url: str, topk: int=100, step: int=None, total_steps: int=None) -> List[float]:
    pattern = r"<rewrite>(.*?)</rewrite>"
    matches = [re.search(pattern, response_str, re.DOTALL) for response_str in response_strs]
    rewards = [0.0 for _ in range(len(matches))]
    valid_rewrites = [match.group(1) for match in matches if match]
    valid_query_ids = [query_ids[i] for i, match in enumerate(matches) if match]
    payload = {
        "queries": valid_rewrites,
        "query_ids": valid_query_ids,
        "topk": topk,
        "return_scores": False
    }
    response = requests.post(retriever_url, json=payload)
    response.raise_for_status()
    rewrite_ranks: List[int] = response.json()['result']
    rank_cnt = 0
    for i, match in enumerate(matches):
        if match:
            rewards[i] = cur_cal_reward(rewrite_ranks[rank_cnt], step, total_steps)
            rank_cnt += 1
    return rewards


def compute_score(
    response_strs: List[str], 
    query_ids: List[str], 
    format_score=0.1, 
    retriever_url='http://127.0.0.1:8000/retrieve',
    topk=100,
    step=None,
    total_steps=None,
):
    if step is not None and total_steps is not None:
        format_rewards = format_reward(response_strs, format_score=format_score)
        rewrite_rewards = cur_rewrite_reward(response_strs, query_ids, retriever_url, topk, step, total_steps)
        return [r1 if r2 == 0.0 else r2 for r1, r2 in zip(rewrite_rewards, format_rewards)]
    else:
        format_rewards = format_reward(response_strs, format_score=format_score)
        rewrite_rewards = rewrite_reward(response_strs, query_ids, retriever_url, topk)
        assert len(rewrite_rewards) == len(format_rewards)
        return [r1 if r2 == 0.0 else r2 for r1, r2 in zip(rewrite_rewards, format_rewards)]
    