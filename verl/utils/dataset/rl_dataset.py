# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """Collate a batch of data."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import (process_image,
                                                         process_video)

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        
        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()


class IndexRLHFDataset(RLHFDataset):
    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(data_files, tokenizer, config, processor)
        # 保存原始数据
        self.full_dataframe = self.dataframe.copy()
        
        # 创建索引到行号的映射，以便高效查找
        self.index_map = self._build_index_map()

    def _build_index_map(self):
        """创建extra_info.index到DataFrame行号的映射"""
        index_map = {}
        # for i, row in self.dataframe.iterrows():
        for i in range(len(self.dataframe)):
            row = self.dataframe.iloc[i]
            if isinstance(row.get('extra_info'), dict) and 'index' in row.get('extra_info', {}):
                index_value = row['extra_info']['index']
                index_map[index_value] = i
            else:
                # 如果没有extra_info.index，使用行号作为索引
                index_map[i] = i
        return index_map
    
    def filter_by_indices(self, indices_to_keep):
        """
        根据extra_info.index筛选数据
        
        Args:
            indices_to_keep: 要保留的extra_info.index列表
        """
        print(f'筛选前数据集大小: {len(self.dataframe)}')
        print(f'索引映射大小: {len(self.index_map)}')
        print(f'要保留的索引数量: {len(indices_to_keep)}')
        
        if not indices_to_keep:
            print("警告: 没有提供有效索引，保持数据集不变")
            return
        
        # 先检查索引类型并打印前几个用于调试
        print(f"要保留的索引类型: {type(indices_to_keep[0])}, 前5个值: {indices_to_keep[:5]}")
        print(f"索引映射键类型: {type(list(self.index_map.keys())[0]) if self.index_map else 'N/A'}")
        
        # 转换indices_to_keep为dataframe行号
        valid_row_indices = []
        missing_indices = []
        
        for idx in indices_to_keep:
            # 尝试不同类型的键
            if idx in self.index_map:
                valid_row_indices.append(self.index_map[idx])
            elif str(idx) in self.index_map:  # 尝试字符串类型
                valid_row_indices.append(self.index_map[str(idx)])
            elif int(idx) in self.index_map:  # 尝试整数类型
                valid_row_indices.append(self.index_map[int(idx)])
            else:
                missing_indices.append(idx)
        
        print(f"找到匹配行索引: {len(valid_row_indices)}/{len(indices_to_keep)}")
        if missing_indices:
            print(f"未找到匹配的索引数量: {len(missing_indices)}")
            if len(missing_indices) < 10:
                print(f"未找到的索引: {missing_indices}")
        
        if not valid_row_indices:
            print("警告: 没有找到匹配的索引，保持数据集不变")
            return
        
        # 确保行索引在有效范围内
        valid_row_indices = [i for i in valid_row_indices if 0 <= i < len(self.dataframe)]
        print(f"有效范围内的行索引数量: {len(valid_row_indices)}")
        
        if not valid_row_indices:
            print("警告: 所有索引都超出范围，保持数据集不变")
            return
        
        # 排序并去重行索引
        valid_row_indices = sorted(list(set(valid_row_indices)))
        
        # 筛选数据
        try:
            filtered_df = self.dataframe.iloc[valid_row_indices]
            print(f"筛选得到行数: {len(filtered_df)}")
            
            # 检查筛选结果
            if len(filtered_df) == 0:
                print("错误: 筛选后数据为空，保持数据集不变")
                return
                
            # 重置索引
            self.dataframe = filtered_df.reset_index(drop=True)
            
            # 验证重置索引后的数据帧
            print(f"重置索引后数据行数: {len(self.dataframe)}")
            
            # 更新索引映射
            old_index_map = self.index_map
            self.index_map = self._build_index_map()
            print(f"更新后索引映射大小: {len(self.index_map)}")
            
            # 验证新的索引映射
            if len(self.index_map) == 0 and len(self.dataframe) > 0:
                print("警告: 新索引映射为空，恢复原始数据和映射")
                self.dataframe = self.full_dataframe.copy()
                self.index_map = old_index_map
                return
                
            print(f'筛选后数据集大小: {len(self.dataframe)}')
        except Exception as e:
            print(f"筛选数据时发生错误: {e}")
            import traceback
            traceback.print_exc()
            print("保持数据集不变")
    
    def reset_dataset(self):
        """重置数据集到初始状态"""
        self.dataframe = self.full_dataframe.copy()
        self.index_map = self._build_index_map()
        print(f'数据集已重置，大小: {len(self.dataframe)}')
    
    def get_available_indices(self):
        """获取当前可用的所有extra_info.index值"""
        return list(self.index_map.keys())
    