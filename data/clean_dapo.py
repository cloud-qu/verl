import pandas as pd
import ast

# 读取原始 parquet 文件
df = pd.read_parquet("/home/quyun/verl/data/dapo-math-17k.parquet")

# 确保 extra_info 是字典（如果是字符串格式需要转化）
def parse_extra_info(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x

df["extra_info"] = df["extra_info"].apply(parse_extra_info)

# 提取 'index' 字段
df["index"] = df["extra_info"].apply(lambda x: x.get("index") if isinstance(x, dict) else None)

# 去重：保留相同 index 的第一条记录
df_dedup = df.drop_duplicates(subset="index", keep="first")

# 删除中间临时列（可选）
df_dedup = df_dedup.drop(columns=["index"])

# 保存为新的 parquet 文件
df_dedup.to_parquet("dapo-math-17k-cleaned.parquet", index=False)

print("清理完毕，新文件已保存为 dapo-math-17k.cleaned.parquet")
