# SPDX-License-Identifier: Apache-2.0
# usage:
# VLLM_USE_V1=1 python examples/offline_inference/data_parallel.py
# we need to have a launcher to create multiple data parallel
# ranks. And each rank will create a vLLM instance to process its own prompts.

# this script is adapted from https://docs.vllm.ai/en/latest/getting_started/examples/data_parallel.html 

import os
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
import argparse
import jsonlines
import re


# for ConvSearch-R1
INSTRUCTION = '''Given a query and its context, you must first think about the reasoning process in the mind to decontextualize the query by resolving \
coreference and omission issues. Then, provide the user with a rewrite that retains its original meaning and is as informative as possible to help \
search engines retrieve relevant documents effectively. The reasoning process and rewrite should be enclosed within <think> </think> and <rewrite> </rewrite> tags, respectively, i.e., \
<think> reasoning process here </think>\n<rewrite> rewrite here </rewrite>.

### Context Begin ###
{context}
### Context End ###

Query: {query}
Rewrite:'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True,
        choices=[
            'DS-R1-Distill-Qwen-7B',
            'Qwen2.5-7B-Instruct',
            'Qwen2.5-7B-Instruct-Rewrite',
            "CoT",
            'Get_SFT',
            "ConvSearch-R1",
            'Instruct'
        ]
    )
    parser.add_argument('--dp_master_ip', type=str, default='127.0.0.1')
    parser.add_argument('--dp_size', type=int, default=8)
    parser.add_argument('--gpus_per_dp_rank', type=int, default=1)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.7)
    args = parser.parse_args()
    return args


def extract_reasoning_content(
    output: str, 
    model_name: str,
    start_token: str='<think>', 
    end_token: str='</think>'
):
    '''for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'''
    if model_name == 'DS-R1-Distill-Qwen-7B':
        try:
            reasoning_content, answer = output.split(end_token)
            return reasoning_content, answer
        except:
            return None
    elif model_name == 'Instruct':
        return None, output
    elif model_name == 'CoT':
        match_rewrite = re.search(r"####\s*(.*)", output)
        if match_rewrite:
            rewrite = match_rewrite.group(1)
            return None, rewrite.strip()
        else:
            return None
    elif model_name == 'Qwen2.5-7B-Instruct-Rewrite' or model_name == 'ConvSearch-R1':
        pattern_rewrite = r"<rewrite>(.*?)</rewrite>"
        pattern_think = r"<think>(.*?)</think>"
        match_rewrite = re.search(pattern_rewrite, output, re.DOTALL)
        match_think = re.search(pattern_think, output, re.DOTALL)
        if match_rewrite and match_think:
            reasoning_content = match_think.group(1)
            rewrite = match_rewrite.group(1)
            return reasoning_content, rewrite
        else:
            return None
    elif model_name == 'Get_SFT':
        pattern_rewrite = r"<rewrite>(.*?)</rewrite>"
        pattern_think = r"<think>(.*?)</think>"
        match_rewrite = re.search(pattern_rewrite, output, re.DOTALL)
        match_think = re.search(pattern_think, output, re.DOTALL)
        if match_rewrite and match_think:
            reasoning_content = match_think.group(1)
            rewrite = match_rewrite.group(1)
            return reasoning_content, rewrite
        else:
            return None
    else:
        raise NotImplementedError


def main(
    dp_size, 
    dp_rank, 
    dp_master_ip, 
    dp_master_port, 
    GPUs_per_dp_rank, 
    model_name_or_path, 
    samples,
    output_path,
    model_name,
    temperature,
    lock,
):
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    # set devices for each dp_rank
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) 
        for i in range(dp_rank * GPUs_per_dp_rank, (dp_rank + 1) * GPUs_per_dp_rank)
    )
    print(f"DP rank {dp_rank} needs to process {len(samples)} prompts")

    # Create a sampling params object.
    # since we are doing data parallel, every rank can have different
    # sampling params. here we set different max_tokens for different
    # ranks for demonstration.
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        # top_k=50,
        max_tokens=4096
    )

    # Create an LLM.
    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=GPUs_per_dp_rank,
        enforce_eager=False,
        # max_model_len=32768,
        gpu_memory_utilization=0.8,
        dtype='bfloat16',
    )
    convs = []
    for sample in samples:
        ctx = []
        for i in range(0, len(sample['context']), 2):
            ctx.append(f'Q{i // 2 + 1}: {sample["context"][i]}')
            ctx.append(f'A{i // 2 + 1}: {sample["context"][i + 1]}')
        convs.append([
            {
                'role': 'user',
                'content': INSTRUCTION.format(
                    context='\n'.join(ctx),
                    query=sample['question']
                )
            }
        ])
    batch_size = 1024
    cnt = 0
    for start in range(0, len(convs), batch_size):
        end = start + batch_size
        batch_convs = convs[start:end]
        batch_samples = samples[start:end]
        batch_outputs = llm.chat(batch_convs, sampling_params, add_generation_prompt=True)
        batch_saved_samples = []
        for idx, output in enumerate(batch_outputs):
            generated_text = output.outputs[0].text
            result = extract_reasoning_content(generated_text, model_name)
            if result:
                reasoning_content, rewrite = result
                batch_samples[idx]['reasoning_content'] = reasoning_content
                batch_samples[idx]['rewrite'] = rewrite
                batch_saved_samples.append(batch_samples[idx])
            else:
                # print(f"Failed to extract reasoning content for prompt: {prompt}")
                cnt += 1
        
        if len(batch_saved_samples) == 0:
            continue
        with lock:
            with jsonlines.open(output_path, 'a') as f:
                f.write_all(batch_saved_samples)
    
    print(f"DP rank {dp_rank} failed to extract reasoning content for {cnt} prompts")


if __name__ == "__main__":
    from multiprocessing import Process, Lock
    self_args = parse_args()
    output_dir = os.path.dirname(self_args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Sample prompts.
    finished_samples = []
    if os.path.exists(self_args.output_path):
        with jsonlines.open(self_args.output_path) as f:
            for line in f:
                finished_samples.append(line['qid'])
    
    samples = []
    with jsonlines.open(self_args.input_path) as f:
        for line in f:
            if line['qid'] in finished_samples:
                continue
            samples.append(line)

    if len(samples) == 0:
        print("No samples to process")
        exit(0)

    dp_master_port = get_open_port()
    lock = Lock()
    procs = []
    for dp_rank in range(self_args.dp_size):
        # with DP, each rank should process different prompts.
        # usually all the DP ranks process a full dataset,
        # and each rank processes a different part of the dataset.
        promtps_per_rank = len(samples) // self_args.dp_size
        start = dp_rank * promtps_per_rank
        if dp_rank == self_args.dp_size - 1:
            end = len(samples)
        else:
            end = start + promtps_per_rank
        if start == end:
            continue
        proc = Process(
            target=main,
            args=(
                self_args.dp_size, 
                dp_rank,
                self_args.dp_master_ip, 
                dp_master_port, 
                self_args.gpus_per_dp_rank,
                self_args.model_name_or_path,
                samples[start:end],
                self_args.output_path,
                self_args.model_name,
                self_args.temperature,
                lock,
            )
        )
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join()
        if proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
