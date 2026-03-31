# scripts/generate_calib_mask.py
import torch
import numpy as np
import argparse
from pathlib import Path

def generate_calib_mask(seq_len: int, top_k: int = 16, output_path: str = "calib_mask.pt"):
    """
    生成一个稀疏注意力 mask，形状 (seq_len, seq_len)，布尔类型。
    策略：采用局部窗口 + 随机采样作为示例（实际可替换为更智能的策略）。
    """
    # 示例：保留局部窗口（前后各 256 帧）+ 随机保留额外 top_k 个远程位置
    window = 256
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window)
        end = min(seq_len, i + window + 1)
        mask[i, start:end] = True
        # 随机增加一些远程连接（模拟基于内容的重要性）
        # 实际生产环境中，你可以用少量样本数据计算真实注意力分布，然后选择 top_k 高频位置
        if top_k > 0:
            indices = list(range(0, start)) + list(range(end, seq_len))
            if indices:
                selected = np.random.choice(indices, min(top_k, len(indices)), replace=False)
                mask[i, selected] = True
    torch.save(mask, output_path)
    print(f"Saved mask to {output_path}, shape: {mask.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=4096, help="序列长度（根据你的实际 token 数设定）")
    parser.add_argument("--top_k", type=int, default=16, help="每个 query 额外保留的远程 key 数量")
    parser.add_argument("--output", type=str, default="calib_mask.pt", help="输出文件路径")
    args = parser.parse_args()
    generate_calib_mask(args.seq_len, args.top_k, args.output)