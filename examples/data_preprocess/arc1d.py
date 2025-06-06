"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import argparse
import os
import re
from random import choice, randint, seed
from typing import List, Tuple

from datasets import Dataset, load_dataset
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs


def make_prefix(dp, template_type):
    train_example = dp['train_examples']
    question = dp['question']
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:
Input: {train_example[0]['input']}
Output: {train_example[0]['output']}

Example 2:
Input: {train_example[1]['input']}
Output: {train_example[1]['output']}

Example 3:
Input: {train_example[2]['input']}
Output: {train_example[2]['output']}

Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Describe how you derived the rule and your overall reasoning process in detail before you submit your answer. """ + \
f"Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> {train_example[0]['output']} </answer>. Your final answer should be just the test output grid itself. " + \
f"""Input: {question}.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\nFind the common rule that maps an input grid to an output grid, given the examples below.
Example 1:
Input: {train_example[0]['input']}
Output: {train_example[0]['output']}
Example 2:
Input: {train_example[1]['input']}
Output: {train_example[1]['output']}
Example 3:
Input: {train_example[2]['input']}
Output: {train_example[2]['output']}
Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Describe how you derived the rule and your overall reasoning process in detail before you submit your answer. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> {train_example[0]['output']} </answer>. Your final answer should be just the test output grid itself.
Input: {question}.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""


    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/verl/data/arc1d')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'arc1d'

    local_data_source = f"{os.environ.get('HOME')}/verl/examples/data_preprocess/arc1d"
    if os.path.exists(local_data_source):
        raw_dataset = load_dataset('parquet', data_files={
            "train": os.path.join(local_data_source, "train.parquet"),
            "test": os.path.join(local_data_source, "test.parquet"),
        })
    else:
        raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')

    train_dataset = raw_dataset['train']
    test_dataset = raw_dataset['test']

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "answer": example['answer'],
                "size": example['size']
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 