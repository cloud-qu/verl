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
from typing import List, Optional, Union

from omegaconf import DictConfig, ListConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset


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
        # self.full_dataframe = self.dataframe.copy()
        
        # 创建索引到行号的映射，以便高效查找
        self.index_map = self._build_index_map()

    def _build_index_map(self):
        """创建extra_info.index到DataFrame行号的映射"""
        index_map = {}
        # for i, row in self.dataframe.iterrows():
        for i in range(len(self.dataframe)):
            row = self.dataframe[i]
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
            filtered_df = self.dataframe[valid_row_indices]
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
            
            # # 验证新的索引映射
            # if len(self.index_map) == 0 and len(self.dataframe) > 0:
            #     print("警告: 新索引映射为空，恢复原始数据和映射")
            #     self.dataframe = self.full_dataframe.copy()
            #     self.index_map = old_index_map
            #     return
                
            print(f'筛选后数据集大小: {len(self.dataframe)}')
        except Exception as e:
            print(f"筛选数据时发生错误: {e}")
            import traceback
            traceback.print_exc()
            print("保持数据集不变")
    
    # def reset_dataset(self):
    #     """重置数据集到初始状态"""
    #     self.dataframe = self.full_dataframe.copy()
    #     self.index_map = self._build_index_map()
    #     print(f'数据集已重置，大小: {len(self.dataframe)}')
    
    def get_available_indices(self):
        """获取当前可用的所有extra_info.index值"""
        return list(self.index_map.keys())
    