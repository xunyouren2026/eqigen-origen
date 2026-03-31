# main.py
from auto_config_adapter import AutoConfigAdapter
from utils import set_seed
from api import app as fastapi_app
from webui import create_webui
from inferencer import Inferencer
from models.diffusion import DiffusionScheduler
from models.lens_controller import LensController
from models.video_encoder import VideoEncoder
from models.audio_encoder import AudioEncoder
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.dit import SpatialTemporalUNet
from models.vae import VideoVAE
from config import Config
import argparse
import torch
import torch.distributed as dist
import os
import sys

# 添加项目根目录到路径，确保能导入 utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 注意：如果 inference 文件夹存在，请确保路径正确
try:
    from inference.long_video_generator import LongVideoGenerator
    from models.long_video_planner import LongVideoPlanner
except ImportError:
    LongVideoGenerator = None
    LongVideoPlanner = None


def main():
    parser = argparse.ArgumentParser(
        description="AI Video Generation System - Ultimate Optimized")
    parser.add_argument(
        "--mode", choices=["train", "infer", "api", "webui"], required=True, help="运行模式")
    parser.add_argument("--config", type=str,
                        default="config.yaml", help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, help="检查点路径")
    parser.add_argument("--prompt", type=str,
                        default="a cat playing with a ball", help="提示词")
    parser.add_argument("--output", type=str,
                        default="output.mp4", help="输出文件路径")
    parser.add_argument("--local_rank", type=int,
                        default=-1, help="分布式训练本地 rank")
    parser.add_argument("--long_script", type=str, help="镜头脚本路径（用于长视频）")
    parser.add_argument("--duration", type=float, default=10.0, help="总时长（秒）")
    parser.add_argument("--fps", type=int, default=24, help="帧率")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="预训练模型标识（HuggingFace ID 或本地路径）")
    parser.add_argument("--resolution", type=str, default="1080p",
                        choices=["256p", "360p", "480p", "720p",
                                 "1080p", "1440p", "4k", "8k"],
                        help="输出视频分辨率")
    parser.add_argument("--width", type=int, default=None, help="自定义宽度（像素）")
    parser.add_argument("--height", type=int, default=None, help="自定义高度（像素）")
    parser.add_argument("--no-auto-config",
                        action="store_true", help="禁用智能显存适配（使用配置文件默认值）")
    # 新增多GPU并行参数
    parser.add_argument(
        "--use_parallel", action="store_true", help="启用多GPU并行生成")
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="使用的GPU数量（0=自动检测）")

    args = parser.parse_args()

    # 1. 加载基础配置
    print(f"正在加载配置文件：{args.config} ...")
    config = Config(args.config)

    # 2. 【核心】智能显存适配
    if not args.no_auto_config and torch.cuda.is_available():
        adapter = AutoConfigAdapter()
        adapter.apply_optimizations(config)
    else:
        if not torch.cuda.is_available():
            print("⚠️ 未检测到 CUDA，将使用 CPU 模式 (速度极慢)。")
        else:
            print("ℹ️ 已禁用智能显存适配，将严格使用配置文件中的默认值。")

    # 3. 分布式初始化
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        config.train.local_rank = args.local_rank
        config.train.distributed = True
        config.train.world_size = dist.get_world_size()
    else:
        config.train.distributed = False
        config.train.local_rank = 0
        config.train.world_size = 1

    device = torch.device(
        f"cuda:{config.train.local_rank}" if torch.cuda.is_available() else "cpu"
    )
    set_seed(42)

    print("正在加载模型组件...")

    # 4. 模型加载逻辑
    model = None
    if args.pretrained:
        try:
            from model_loader import load_pretrained_model
            model = load_pretrained_model(config, device,
                                          pretrained_name=args.pretrained,
                                          model_type=config.model.model_type)
        except Exception as e:
            print(f"❌ 加载预训练模型失败：{e}，将回退到随机初始化。")

    if model is None:
        default_model_path = "./models/model.safetensors"
        if os.path.exists(default_model_path):
            print(f"ℹ️ 未指定 --pretrained，发现默认模型：{default_model_path}")
            try:
                from model_loader import load_pretrained_model
                model = load_pretrained_model(config, device,
                                              pretrained_name=default_model_path,
                                              model_type=config.model.model_type)
            except Exception as e:
                print(f"❌ 加载默认模型失败：{e}")

        if model is None:
            print("⚠️ 未找到任何预训练模型，将随机初始化模型（生成效果将极差，仅用于测试流程）。")
            if config.model.mode == 'creative' or args.duration < 30:
                try:
                    from models.simple_dit import SimpleDiT
                    model = SimpleDiT(config.model).to(device)
                except ImportError:
                    model = SpatialTemporalUNet(config.model).to(device)
            else:
                model = SpatialTemporalUNet(config.model).to(device)

    # 5. 初始化其他组件
    vae = VideoVAE(config.model, device=device)
    scheduler = DiffusionScheduler(config.model, device)
    text_encoder = TextEncoder(
        config.model.text_encoder, config.model.text_encoder_model, device
    )

    image_encoder = ImageEncoder(
        config.model.image_encoder_model, device
    ) if config.model.use_image_encoder else None

    audio_encoder = AudioEncoder(
        config.model.audio_encoder_model, device
    ) if config.model.use_audio_encoder else None

    video_encoder = VideoEncoder(
        vae, config.model
    ) if config.model.use_video_encoder else None

    lens_controller = LensController(config.model)

    print("✅ 模型组件加载完成！")
    print(f"   当前配置：BlockFrames={config.model.max_block_frames}, "
          f"MemorySize={config.model.memory_size}, VAE={config.model.vae_type}, "
          f"TileBatch={config.model.tile_batch_size}")

    # 6. 执行对应模式
    if args.mode == "train":
        from trainer import Trainer
        from data.dataset import get_dataloader

        print("准备训练数据...")
        train_loader = get_dataloader(config.data, split="train")
        val_loader = get_dataloader(config.data, split="val")

        trainer = Trainer(config, model, vae, scheduler, text_encoder,
                          image_encoder, audio_encoder, video_encoder,
                          lens_controller, device)

        if args.checkpoint:
            print(f"从检查点加载：{args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)

        print("开始训练...")
        trainer.train(train_loader, val_loader)

    elif args.mode == "infer":
        inferencer = Inferencer(config, model, vae, scheduler, text_encoder,
                                image_encoder, audio_encoder, video_encoder,
                                lens_controller, device)

        if args.checkpoint:
            print(f"从检查点加载权重：{args.checkpoint}")
            state = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state['model'], strict=False)
            vae.load_state_dict(state['vae'], strict=False)
            # 可选加载其他编码器权重...

        output_final = args.output

        if args.long_script and LongVideoGenerator and LongVideoPlanner:
            print(f"使用镜头脚本生成长视频：{args.long_script}")
            planner = LongVideoPlanner(
                block_duration_sec=config.long_video.block_duration_sec,
                fps=args.fps,
                overlap_sec=config.long_video.overlap_sec
            )
            generator = LongVideoGenerator(
                inferencer, planner, num_workers=config.long_video.num_workers
            )
            video = generator.generate(
                args.long_script, args.duration, args.prompt,
                output_format=output_final.split('.')[-1], fps=args.fps
            )
            from utils import save_video
            save_video(video, output_final, fps=args.fps)
        else:
            print(
                f"开始生成视频：'{args.prompt}' ({args.duration}s, {args.resolution})...")
            if args.duration < 30:
                # 短视频：创意模式
                inferencer.generate(
                    args.prompt, output_path=output_final,
                    duration=args.duration, fps=args.fps,
                    resolution=args.resolution,
                    width=args.width, height=args.height
                )
            else:
                # 长视频：自动选择优化等级
                optimization_level = 'full' if args.duration > 300 else 'light'
                output_path, _ = inferencer.generate_long(
                    prompt=args.prompt,
                    duration=args.duration,
                    fps=args.fps,
                    resolution=args.resolution,
                    width=args.width,
                    height=args.height,
                    cfg_scale=config.model.cfg_scale,
                    steps=config.model.num_inference_steps,  # 注意：如果开启了蒸馏，WebUI 或内部逻辑会覆盖此值
                    negative_prompt="",
                    optimization_level=optimization_level,
                    use_parallel=args.use_parallel,      # 新增
                    num_gpus=args.num_gpus               # 新增
                )
                output_final = output_path

        print(f"🎉 视频已成功保存到：{output_final}")

    elif args.mode == "api":
        print("初始化 API 服务...")
        inferencer = Inferencer(config, model, vae, scheduler, text_encoder,
                                image_encoder, audio_encoder, video_encoder,
                                lens_controller, device)
        fastapi_app.state.inferencer = inferencer
        fastapi_app.state.results = {}
        fastapi_app.state.tasks = {}
        os.makedirs(config.api.temp_dir, exist_ok=True)

        print(f"🚀 API 服务启动中...")
        print(f"   访问地址：http://{config.api.host}:{config.api.port}")
        print(f"   API 文档：http://{config.api.host}:{config.api.port}/docs")

        import uvicorn
        uvicorn.run(fastapi_app, host=config.api.host, port=config.api.port)

    elif args.mode == "webui":
        print("启动 WebUI...")
        inferencer = Inferencer(config, model, vae, scheduler, text_encoder,
                                image_encoder, audio_encoder, video_encoder,
                                lens_controller, device)
        create_webui(inferencer)

    else:
        print("❌ 未知模式")


if __name__ == "__main__":
    main()
