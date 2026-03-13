from datasets import load_dataset

# 1. データセットの読み込み
# (このデータセットは構造によって'train'などのsplit指定が必要な場合があります)
dataset = load_dataset("jianzongwu/MangaZero", split="train")

# 2. 基本情報の確認
print("--- データセットの構造 ---")
print(dataset)
print("\n--- カラム名 ---")
print(dataset.column_names)

# 3. 最初の1件を表示して中身を確認
print("\n--- データのサンプル (1件目) ---")
sample = dataset[0]
for key, value in sample.items():
    # データが長すぎる場合は一部だけ表示
    content = str(value)
    if len(content) > 200:
        content = content[:200] + "..."
    print(f"[{key}]: {content}")

# 4. (オプション) Pandas DataFrame形式で綺麗に表示
import pandas as pd

df = dataset.select(range(5)).to_pandas()  # 最初の5件を抽出
print("\n--- テーブル形式での表示 (先頭5件) ---")
print(df)
