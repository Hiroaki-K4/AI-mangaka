import json
import os
from io import BytesIO

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import requests
from datasets import load_dataset
from PIL import Image

# 出力ディレクトリの作成
output_dir = "visualized_samples"
os.makedirs(output_dir, exist_ok=True)

# データセットの読み込み
print("データセットを読み込んでいます...")
dataset = load_dataset("jianzongwu/MangaZero", split="train")

# 可視化する画像の数
num_samples = 5

print(f"\n{num_samples}枚の画像を可視化します...")

for idx in range(num_samples):
    sample = dataset[idx]
    image_path = sample["image_path"]
    frames = sample["frames"]
    meta = sample["meta"]

    print(f"\n[{idx+1}/{num_samples}] {image_path} を処理中...")

    # 画像の読み込み（URLから）
    try:
        # url1を使用して画像をダウンロード
        url = meta["url1"]
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))

        # RGB変換（必要に応じて）
        if img.mode != "RGB":
            img = img.convert("RGB")

        # プロット作成
        fig, ax = plt.subplots(1, figsize=(20, 15))
        ax.imshow(img)

        # テキスト情報を保存するリスト
        text_info = []
        text_info.append(f"Image: {image_path}")
        text_info.append(f"Image Size: {img.size}")
        text_info.append(
            f"Meta: url1={meta['url1'][:60]}..., width1={meta.get('width1', 'N/A')}"
        )
        if "url2" in meta:
            text_info.append(
                f"      url2={meta['url2'][:60]}..., width2={meta.get('width2', 'N/A')}"
            )
        text_info.append(f"\nTotal Frames: {len(frames)}\n")

        # 各フレーム（バウンディングボックス）を描画
        for frame_idx, frame in enumerate(frames):
            bbox = frame["bbox"]
            caption = frame["caption"]
            characters = frame.get("characters", [])
            dialogs = frame.get("dialogs", [])

            # bboxの形式を確認（4つの値）
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            # フレームのバウンディングボックスを描画（太線）
            colors = [
                "red",
                "blue",
                "green",
                "yellow",
                "cyan",
                "magenta",
                "orange",
                "purple",
            ]
            color = colors[frame_idx % len(colors)]

            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=3,
                edgecolor=color,
                facecolor="none",
                linestyle="-",
                label=f"Frame {frame_idx+1}",
            )
            ax.add_patch(rect)

            # フレーム番号を表示
            ax.text(
                x1,
                y1 - 10,
                f"F{frame_idx+1}",
                color="white",
                fontsize=12,
                weight="bold",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
            )

            # ダイアログのバウンディングボックスを描画（細線、点線）
            for dialog_idx, dialog in enumerate(dialogs):
                if "bbox" in dialog:
                    d_bbox = dialog["bbox"]
                    dx1, dy1, dx2, dy2 = d_bbox
                    d_width = dx2 - dx1
                    d_height = dy2 - dy1

                    d_rect = patches.Rectangle(
                        (dx1, dy1),
                        d_width,
                        d_height,
                        linewidth=2,
                        edgecolor=color,
                        facecolor="none",
                        linestyle="--",
                        alpha=0.7,
                    )
                    ax.add_patch(d_rect)

            # テキスト情報を収集
            text_info.append(f"--- Frame {frame_idx+1} ---")
            text_info.append(f"BBox: {bbox}")
            text_info.append(f"Caption: {caption}")
            text_info.append(f"Characters: {characters if characters else '[]'}")
            text_info.append(f"Dialogs: {len(dialogs)} dialog boxes")
            for dialog_idx, dialog in enumerate(dialogs):
                text_info.append(f"  Dialog {dialog_idx+1}: {dialog}")
            text_info.append("")

        ax.axis("off")
        title = f"{image_path}\n({len(frames)} frames, "
        total_dialogs = sum(len(f.get("dialogs", [])) for f in frames)
        title += f"{total_dialogs} dialogs)"
        plt.title(title, fontsize=16)

        # 凡例を追加
        legend_elements = [
            patches.Patch(
                facecolor="none", edgecolor="black", linewidth=3, label="Frame (実線)"
            ),
            patches.Patch(
                facecolor="none",
                edgecolor="black",
                linewidth=2,
                linestyle="--",
                label="Dialog (点線)",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

        plt.tight_layout()

        # 画像を保存
        output_filename = (
            f"{output_dir}/sample_{idx:03d}_{image_path.replace('/', '_')}.png"
        )
        plt.savefig(output_filename, dpi=150, bbox_inches="tight")
        plt.close()

        # テキスト情報を保存
        text_filename = (
            f"{output_dir}/sample_{idx:03d}_{image_path.replace('/', '_')}.txt"
        )
        with open(text_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(text_info))

        print(f"  ✓ 画像保存: {output_filename}")
        print(f"  ✓ テキスト保存: {text_filename}")
        print(f"  - フレーム数: {len(frames)}")
        print(f"  - ダイアログ数: {total_dialogs}")
        print(f"  - 画像サイズ: {img.size}")

    except Exception as e:
        print(f"  ✗ エラー: {e}")
        import traceback

        traceback.print_exc()
        continue

print(f"\n完了！ {output_dir}/ ディレクトリに画像とテキストファイルが保存されました。")
