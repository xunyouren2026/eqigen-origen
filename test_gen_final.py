import sys, traceback, torch, os
from config import Config
from models.vae import VideoVAE
from models.dit import SpatialTemporalUNet
from models.text_encoder import TextEncoder
from models.diffusion import DiffusionScheduler
from inferencer import Inferencer

def main():
    print("=== 开始测试生成 ===")
    try:
        config = Config()
        config.model.enable_teacache = False
        device = torch.device("cpu")
        print(f"使用设备: {device}")

        print("加载模型...")
        model = SpatialTemporalUNet(config.model).to(device)
        vae = VideoVAE(config.model, device)
        scheduler = DiffusionScheduler(config.model, device)
        text_encoder = TextEncoder(config.model.text_encoder, config.model.text_encoder_model, device)

        print("初始化 Inferencer...")
        inferencer = Inferencer(config, model, vae, scheduler, text_encoder, None, None, None, None, device)

        output_path = "D:\\video_gen\\test_output.mp4"
        duration = 1
        fps = 24
        num_steps = 2          # 只用2步，快速
        print(f"开始生成视频 (时长={duration}s, fps={fps}, 步数={num_steps})，保存到: {output_path}")
        result = inferencer.generate(
            prompt="a cat",
            duration=duration,
            fps=fps,
            resolution="256p",
            num_steps=num_steps,
            output_path=output_path
        )
        print(f"生成完成！返回路径: {result}")

        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"文件大小: {size} 字节")
            if size > 1024:
                print("✅ 视频文件已成功生成！")
            else:
                print("⚠️ 文件大小异常，可能生成失败。")
        else:
            print("❌ 未找到输出文件！")
    except Exception as e:
        print("发生错误:")
        traceback.print_exc()
    input("按回车键退出...")

if __name__ == "__main__":
    main()