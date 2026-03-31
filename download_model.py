# download_model.py
from huggingface_hub import hf_hub_download, snapshot_download
import os
import argparse


def download_wan_model():
    os.makedirs("./models", exist_ok=True)
    # 下载万相2.1 1.3B模型（约 5GB）
    model_path = hf_hub_download(
        repo_id="Wan-AI/Wan2.1-1.3B",
        filename="model.safetensors",
        local_dir="./models"
    )
    print(f"Wan2.1模型下载到: {model_path}")


def download_lean_vae():
    """下载 LeanVAE 权重"""
    os.makedirs("./models/LeanVAE", exist_ok=True)
    snapshot_download(
        repo_id="westlake-repl/LeanVAE",
        local_dir="./models/LeanVAE",
        ignore_patterns=["*.bin", "*.pth"]  # 根据需要调整
    )
    print("LeanVAE 下载到 ./models/LeanVAE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载预训练模型")
    parser.add_argument(
        "--model", choices=["wan", "lean_vae", "all"], default="all", help="选择要下载的模型")
    args = parser.parse_args()

    if args.model in ["wan", "all"]:
        download_wan_model()
    if args.model in ["lean_vae", "all"]:
        download_lean_vae()
