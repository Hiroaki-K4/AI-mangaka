import json

from datasets import load_dataset

# データセットの読み込み
print("データセットを読み込んでいます...")
dataset = load_dataset("jianzongwu/MangaZero", split="train")

# 最初のサンプルの完全な構造を確認
sample = dataset[0]

print("\n=== 完全なデータ構造 ===\n")

print("1. image_path:")
print(f"   {sample['image_path']}\n")

print("2. frames (最初の1つのみ詳細表示):")
if len(sample["frames"]) > 0:
    print(f"   フレーム数: {len(sample['frames'])}")
    print(f"   最初のフレームのキー: {sample['frames'][0].keys()}")
    print(f"   詳細:")
    for key, value in sample["frames"][0].items():
        print(f"     - {key}: {value}")
    print()

print("3. meta:")
print(f"   metaのキー: {sample['meta'].keys()}")
print(f"   詳細:")
for key, value in sample["meta"].items():
    content = str(value)
    if len(content) > 100:
        content = content[:100] + "..."
    print(f"     - {key}: {content}")
print()

# すべてのカラムを確認
print("=== すべてのカラム ===")
print(f"データセットのカラム: {dataset.column_names}")
print()

# 複数のサンプルでframesの構造が一貫しているか確認
print("=== framesの構造の一貫性確認 (最初の5件) ===")
for i in range(min(5, len(dataset))):
    sample = dataset[i]
    if len(sample["frames"]) > 0:
        frame_keys = list(sample["frames"][0].keys())
        print(f"サンプル {i}: フレーム数={len(sample['frames'])}, キー={frame_keys}")
