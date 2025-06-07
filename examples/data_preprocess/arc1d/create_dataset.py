import json
import os

import pyarrow.parquet as pq
import reasoning_gym
from datasets import Dataset


def create_and_save_dataset(name, min_size, max_size, num_train, size, seed, split, output_dir):
    # 创建 reasoning_gym 数据集
    data = reasoning_gym.create_dataset(name,
                                        min_size=min_size,
                                        max_size=max_size,
                                        num_train=num_train,
                                        seed=seed,
                                        size=size)

    # 转为 list[dict]
    data_list = [{
        # "question": x["question"],
        # "answer": x["answer"],
        "train_examples": x["metadata"]["train_examples"],
        "size": x["metadata"]["size"],
        "question": x["metadata"]["test_example"]["input"],
        "answer": x["metadata"]["test_example"]["output"],
    } for x in data]

    # 校验答案正确性
    for entry in data:
        assert data.score_answer(entry["answer"], entry) == 1.0

    # 转为 Hugging Face Dataset
    hf_dataset = Dataset.from_list(data_list)
    # hf_dataset.save_to_disk(f"{output_dir}/{split}")

    # 保存为 Parquet 文件
    os.makedirs(output_dir, exist_ok=True)
    import pyarrow as pa
    table = pa.Table.from_batches(hf_dataset.data.to_batches())
    pq.write_table(table, f"{output_dir}/{split}.parquet")
    # pq.write_table(hf_dataset.data, f"{output_dir}/{split}.parquet")
    json_path = os.path.join(output_dir, f"{split}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)



# 创建并保存 train 集
create_and_save_dataset(
    name="arc_1d",
    min_size=10,
    max_size=20,
    num_train=3,
    size=2000,
    seed=42,
    split="train",
    output_dir="data/arc1d"
)

# 创建并保存 test 集
create_and_save_dataset(
    name="arc_1d",
    min_size=10,
    max_size=20,
    num_train=3,
    size=512,
    seed=42,
    split="test",
    output_dir="data/arc1d"
)
