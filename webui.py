# webui.py
import gradio as gr
import numpy as np
import cv2
from inferencer import Inferencer
import json
import tempfile
import os
from script_generator import ScriptGenerator


def create_webui(inferencer: Inferencer):
    with gr.Blocks(title="AI视频工坊 · 专业版", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 AI视频生成系统 · 专业版
        支持自回归长视频生成、记忆注入、4K/8K超分、蒸馏加速等前沿技术
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # 基础输入
                prompt = gr.Textbox(
                    label="✨ 提示词",
                    placeholder="例如：a cat playing with a ball in a garden",
                    lines=2
                )
                negative = gr.Textbox(
                    label="⚠️ 负面提示词",
                    placeholder="blur, low quality, distortion",
                    lines=1
                )

                with gr.Row():
                    duration = gr.Slider(
                        1, 300, value=5, step=1, label="⏱️ 时长 (秒)")
                    fps = gr.Slider(1, 120, value=24, step=1,
                                    label="🎞️ 帧率 (fps)")

                with gr.Row():
                    cfg_scale = gr.Slider(
                        1, 15, value=7.5, step=0.1, label="🎚️ CFG 强度")
                    steps = gr.Slider(1, 100, value=50, step=1, label="🔢 推理步数")

                with gr.Row():
                    resolution = gr.Dropdown(
                        choices=["256p", "360p", "480p", "720p",
                                 "1080p", "1440p (2K)", "4K", "8K"],
                        value="1080p",
                        label="📺 输出分辨率"
                    )
                    output_format = gr.Dropdown(
                        choices=["mp4", "gif", "webm"],
                        value="mp4",
                        label="📁 输出格式"
                    )

                # 高级选项折叠面板
                with gr.Accordion("⚙️ 高级生成设置", open=False):
                    with gr.Row():
                        ar_mode = gr.Checkbox(
                            label="🔄 自回归生成 (逐帧预测，长视频专用)", value=False)
                        use_memory = gr.Checkbox(
                            label="🧠 记忆注入 (保持角色/场景一致)", value=True)
                        distill_mode = gr.Checkbox(
                            label="⚡ 蒸馏模式 (4步快速推理，需蒸馏模型)", value=False)
                        use_raft = gr.Checkbox(
                            label="🌊 物理约束增强 (需RAFT光流)", value=False)
                    with gr.Row():
                        use_routing = gr.Checkbox(
                            label="🎯 动态路由 (增强长程关联)", value=False)
                        use_compression = gr.Checkbox(
                            label="📦 分层压缩 (减少显存)", value=False)
                        use_pyramid = gr.Checkbox(
                            label="🏔️ 金字塔采样 (加速高分辨率)", value=False)
                    with gr.Row():
                        style = gr.Dropdown(
                            choices=["无", "cartoon", "oil", "sketch",
                                     "sepia", "pixelate", "edge"],
                            value="无",
                            label="🎨 风格化"
                        )
                        camera = gr.Dropdown(
                            choices=["无", "pan", "tilt", "zoom",
                                     "rotate", "dolly", "track"],
                            value="无",
                            label="🎥 镜头运动"
                        )
                    with gr.Row():
                        use_learned_compressor = gr.Checkbox(
                            label="🔧 可学习压缩器 (联合训练)", value=False)
                        use_concatenated_history = gr.Checkbox(
                            label="🔗 拼接历史注入", value=False)
                    watermark = gr.Textbox(
                        label="💧 水印文字", placeholder="AI Video")
                    pretrained_path = gr.Textbox(
                        label="🔗 预训练模型路径 (本地或 HuggingFace ID)",
                        placeholder="./models/model.safetensors 或 Wan-AI/Wan2.1-1.3B",
                        value="./models/model.safetensors"
                    )

                # 多模态输入区
                with gr.Accordion("📎 多模态输入", open=False):
                    reference_images = gr.File(label="参考图片 (可多张)", file_types=[
                                               "image"], file_count="multiple")
                    reference_videos = gr.File(label="参考视频 (可多段)", file_types=[
                                               "video"], file_count="multiple")
                    audio_files = gr.File(label="音频文件 (可多段)", file_types=[
                                          "audio"], file_count="multiple")

                    with gr.Row():
                        story_input = gr.Textbox(
                            label="📝 故事创意",
                            lines=2,
                            placeholder="输入故事梗概，将自动生成镜头脚本...",
                            scale=3
                        )
                        generate_script_btn = gr.Button(
                            "✨ 自动生成脚本", variant="secondary", scale=1)
                    lens_script = gr.File(
                        label="镜头脚本 (JSON)", file_types=[".json"])

                # 设备提示
                device_type = inferencer.device.type
                if device_type == 'cuda':
                    device_msg = "✅ GPU 可用，推荐使用高分辨率生成"
                else:
                    device_msg = "⚠️ CPU 模式，请降低分辨率/帧率/时长，否则可能内存不足"

                device_info = gr.HTML(f"""
                <div style="background: #f0f0f0; padding: 8px; border-radius: 4px; margin-top: 10px;">
                💡 当前运行设备: <strong>{device_type.upper()}</strong><br>
                {device_msg}
                </div>
                """)

                btn = gr.Button("🚀 开始生成视频", variant="primary")

            with gr.Column(scale=1):
                video_output = gr.Video(label="生成视频", show_label=True)

        # 脚本生成回调函数
        def on_generate_script(story):
            if not story or not story.strip():
                gr.Warning("请先输入故事创意")
                return None
            try:
                # 使用本地模型或 API（根据实际情况调整）
                generator = ScriptGenerator(use_local=True)  # 确保本地模型已下载
                script_dict = generator.generate(story)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                    json.dump(script_dict, f, indent=2, ensure_ascii=False)
                    temp_path = f.name
                return temp_path
            except Exception as e:
                gr.Error(f"脚本生成失败: {str(e)}")
                return None

        generate_script_btn.click(
            fn=on_generate_script,
            inputs=story_input,
            outputs=lens_script
        )

        # 生成函数包装器
        def generate_wrapper(
            prompt, negative, duration, fps, cfg_scale, steps,
            resolution, output_format, ar_mode, use_memory, distill_mode, use_raft,
            style, camera, watermark, pretrained_path,
            reference_images, reference_videos, audio_files, lens_script,
            use_routing, use_compression, use_pyramid
        ):
            if distill_mode and steps > 4:
                steps = 4
                print("蒸馏模式已启用，步数强制设为4")

            # 分辨率映射
            res_map = {
                "256p": (256, 256), "360p": (640, 360), "480p": (854, 480),
                "720p": (1280, 720), "1080p": (1920, 1080), "1440p (2K)": (2560, 1440),
                "4K": (3840, 2160), "8K": (7680, 4320),
            }
            target_w, target_h = res_map.get(resolution, (256, 256))

            style_val = None if style == "无" else style
            camera_val = None if camera == "无" else camera

            ref_imgs = []
            if reference_images:
                for file in reference_images:
                    img = cv2.imread(file.name)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        ref_imgs.append(img)

            ref_vids = []
            if reference_videos:
                for file in reference_videos:
                    cap = cv2.VideoCapture(file.name)
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cap.release()
                    if frames:
                        ref_vids.append(np.array(frames))

            audio_paths = [
                f.name for f in audio_files] if audio_files else None
            lens_path = lens_script.name if lens_script else None

            # 临时设置配置（生产环境应通过参数传递）
            inferencer.config.model.use_token_routing = use_routing
            inferencer.config.model.use_hierarchical_compression = use_compression
            inferencer.config.model.use_pyramid_sampling = use_pyramid

            try:
                if ar_mode:
                    if not hasattr(inferencer, 'generate_ar'):
                        raise AttributeError("当前 inferencer 不支持自回归生成，请更新代码")
                    output_path = inferencer.generate_ar(
                        prompt=prompt,
                        duration=duration,
                        fps=fps,
                        cfg_scale=cfg_scale,
                        steps=steps,
                        negative_prompt=negative,
                        use_memory=use_memory,
                        output_path=None
                    )
                else:
                    output_path, _ = inferencer.generate_long(
                        prompt=prompt,
                        duration=duration,
                        fps=fps,
                        width=target_w,
                        height=target_h,
                        cfg_scale=cfg_scale,
                        steps=steps,
                        negative_prompt=negative,
                        style=style_val,
                        camera=camera_val,
                        output_format=output_format,
                        watermark=watermark,
                        reference_images=ref_imgs if ref_imgs else None,
                        reference_videos=ref_vids if ref_vids else None,
                        audio_paths=audio_paths,
                        lens_script_path=lens_path
                    )
                return output_path
            except Exception as e:
                gr.Error(f"生成失败: {str(e)}")
                return None

        btn.click(
            fn=generate_wrapper,
            inputs=[
                prompt, negative, duration, fps, cfg_scale, steps,
                resolution, output_format, ar_mode, use_memory, distill_mode, use_raft,
                style, camera, watermark, pretrained_path,
                reference_images, reference_videos, audio_files, lens_script,
                use_routing, use_compression, use_pyramid
            ],
            outputs=video_output
        )

        def set_steps_for_distill(distill):
            if distill:
                return gr.update(value=4, maximum=4, interactive=False)
            else:
                return gr.update(value=50, maximum=100, interactive=True)

        distill_mode.change(fn=set_steps_for_distill, inputs=[
                            distill_mode], outputs=[steps])

        gr.HTML("""
        <div style="background: #fff3cd; padding: 10px; border-radius: 4px; margin-top: 20px;">
        ⚠️ 重要提示：
        <ul>
        <li>首次使用请下载预训练模型（约5GB），并在“预训练模型路径”中指定。</li>
        <li>CPU模式生成速度极慢，建议使用NVIDIA GPU（显存≥8GB）。</li>
        <li>自回归生成需要显存≥16GB，否则可能OOM。</li>
        <li>蒸馏模式需要模型本身经过蒸馏训练（如LCM/DMD），否则低步数会导致画质严重下降。</li>
        <li>4K/8K生成会自动分块，但显存仍需≥16GB。</li>
        </ul>
        </div>
        """)

    demo.launch(server_port=7860, inbrowser=True)


if __name__ == "__main__":
    # 仅用于演示，实际应从外部传入 inferencer
    pass
