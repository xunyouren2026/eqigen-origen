# inferencer.py
# AI 视频生成系统 - 推理核心模块 (Ultimate Optimized Version)
# 功能：集成 TeaCache, FP8, TensorRT, 智能分块，轻度物理后处理
# 状态：✅ 已修复 CPU/GPU 数据类型兼容性问题
#      ✅ 已修复模型返回值个数不匹配问题（兼容 SimpleDiT 和 SpatialTemporalUNet）
#      ✅ 已修复创意模式中传递多余关键字参数的问题
#      ✅ 新增 TensorRT 完整推理实现
#      ✅ 新增流水线并行生成方法
#      ✅ 新增 NaN 检测与数值稳定（clamp）

import torch
import numpy as np
import cv2
import sys
import time
import os
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# 项目内部模块
from postprocess import postprocess_video
from utils import save_video
from models.lens_controller import LensScriptParser
from postprocess.tile_generator import TileGenerator
from models.memory_bank import MemoryBank
from postprocess.frame_interpolation import RIFEInterpolator
from models.trainable_memory import LightweightMemory
from models.memory_fusion import MemoryFusion
from teacache import TeaCache


class Inferencer:
    def __init__(self, config, model, vae, scheduler, text_encoder,
                 image_encoder, audio_encoder, video_encoder, lens_controller, device):
        self.config = config
        self.model = model
        self.vae = vae
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.lens_controller = lens_controller
        self.device = device
        self.last_state = None

        # 模式标识（creative / long）
        self.mode = config.model.mode

        # 特征缓存
        self.text_cache = {}
        self.max_cache_size = 100

        # 初始化 tile 生成器
        self.tile_generator = TileGenerator(self, tile_size=512, overlap=32,
                                            batch_size=config.model.tile_batch_size)

        # ========== 根据设备选择精度 ==========
        self.use_half = (device.type == 'cuda')  # GPU 用半精度，CPU 用全精度
        print(
            f"[Inferencer] 使用 {'半精度 (FP16)' if self.use_half else '全精度 (FP32)'}")

        # 原模型准备（PyTorch）
        if self.use_half:
            self.orig_model = model.to(device).half()
        else:
            self.orig_model = model.to(device).float()
        self.orig_model.eval()

        # ========== NVFP4/FP8 量化 (仅 GPU) ==========
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
            try:
                self.orig_model = self.orig_model.to(torch.float8_e4m3fn)
                print("[NVFP8] Model converted to FP8 (e4m3fn) for faster inference")
            except Exception as e:
                print(
                    f"[NVFP8] FP8 conversion failed: {e}, falling back to FP16")
        else:
            print("[NVFP8] Hardware does not support FP8, using FP16")

        # ========== TensorRT 加载 ==========
        self.use_tensorrt = getattr(config.model, 'use_tensorrt', False)
        if self.use_tensorrt:
            self.model = self._load_tensorrt_engine(
                config.model.tensorrt_engine_path)
            if self.model is None:
                print(
                    "[WARN] TensorRT engine not found or load failed, falling back to PyTorch")
                self.model = self.orig_model
                self.use_tensorrt = False
        else:
            self.model = self.orig_model

        # 轨迹投影层
        self.traj_proj = nn.Linear(
            3, self.config.model.dit_context_dim).to(device)
        if self.use_half:
            self.traj_proj = self.traj_proj.half()

        # ========== 内部记忆初始化 ==========
        self.internal_memory = None
        if self.mode == 'long' and self.config.model.use_internal_memory:
            self.internal_memory = LightweightMemory(
                num_slots=self.config.model.memory_slots,
                dim=self.config.model.dit_context_dim,
                rank=self.config.model.memory_rank,
                top_k=self.config.model.memory_top_k,
                controller_type=self.config.model.memory_controller
            ).to(device)
            if self.use_half:
                self.internal_memory = self.internal_memory.half()

            memory_fusion = MemoryFusion(
                dim=self.config.model.dit_context_dim,
                num_heads=self.config.model.dit_num_heads,
                use_linear=self.config.model.use_linear_fusion
            ).to(device)
            if self.use_half:
                memory_fusion = memory_fusion.half()

            for block in self.model.blocks:
                block.memory_fusion = memory_fusion
                block.use_internal_memory = True

        # ========== TeaCache 初始化 ==========
        self.enable_teacache = getattr(config.model, 'enable_teacache', False)
        if self.enable_teacache:
            self.tea_cache = TeaCache(
                threshold=getattr(config.model, 'teacache_threshold', 0.15),
                model=self.model,
                device=self.device
            )
            print(
                f"[TeaCache] Enabled with threshold {self.tea_cache.threshold}")
        else:
            self.tea_cache = None

        # ========== 蒸馏模型检测与步数修正 ==========
        self._check_distillation_compatibility()

    def _check_distillation_compatibility(self):
        """检查当前模型是否支持蒸馏模式，若不支持则自动调整步数并警告"""
        if self.config.model.num_inference_steps < 50:
            # 检测模型是否具有蒸馏特征（例如 flow_head 或 dual_stream 模块）
            has_distilled_feature = False
            for name, _ in self.model.named_modules():
                if 'flow_head' in name or 'dual_stream' in name or 'head' in name:
                    has_distilled_feature = True
                    break
            if not has_distilled_feature:
                print("[WARNING] 蒸馏模式已开启，但模型似乎未经过蒸馏训练，将自动将推理步数恢复为 50。")
                self.config.model.num_inference_steps = 50

    def _load_tensorrt_engine(self, engine_path):
        """加载 TensorRT 引擎并返回一个可调用的包装器（完整实现）"""
        if not os.path.exists(engine_path):
            return None
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            print("[ERROR] TensorRT or pycuda not installed.")
            return None

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            return None
        context = engine.create_execution_context()

        # 获取输入输出名称和形状信息
        input_names = []
        output_names = []
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            if engine.binding_is_input(i):
                input_names.append(name)
            else:
                output_names.append(name)

        # 分配 GPU 内存（使用固定缓冲区，假设输入形状固定；若需动态形状，可后续调整）
        input_buffers = {}
        output_buffers = {}
        for name in input_names:
            shape = tuple(engine.get_binding_shape(name))
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_binding_dtype(name))
            buffer = cuda.mem_alloc(size * dtype().itemsize)
            input_buffers[name] = {'buffer': buffer,
                                   'shape': shape, 'dtype': dtype}
        for name in output_names:
            shape = tuple(engine.get_binding_shape(name))
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_binding_dtype(name))
            buffer = cuda.mem_alloc(size * dtype().itemsize)
            output_buffers[name] = {'buffer': buffer,
                                    'shape': shape, 'dtype': dtype}

        class TensorRTModel:
            def __init__(self, engine, context, input_buffers, output_buffers, input_names, output_names):
                self.engine = engine
                self.context = context
                self.input_buffers = input_buffers
                self.output_buffers = output_buffers
                self.input_names = input_names
                self.output_names = output_names

            def __call__(self, x, t, cond, **kwargs):
                """
                执行 TensorRT 推理
                x: 潜变量 (B, C, T, H, W) torch.Tensor
                t: 时间步 (B,) torch.Tensor
                cond: 条件嵌入 (B, L, D) torch.Tensor
                返回: (noise_pred, flow_pred, new_state) 与原始模型接口一致
                """
                # 假设引擎输入顺序为 ['input', 'timestep', 'context']，输出为 ['output']
                # 实际名称可能不同，这里按索引取，但使用名称更健壮
                # 将输入数据展平并拷贝到 GPU 缓冲区
                x_cpu = x.cpu().numpy().ravel()
                t_cpu = t.cpu().numpy().ravel()
                cond_cpu = cond.cpu().numpy().ravel()

                cuda.memcpy_htod(
                    self.input_buffers[self.input_names[0]]['buffer'], x_cpu)
                cuda.memcpy_htod(
                    self.input_buffers[self.input_names[1]]['buffer'], t_cpu)
                cuda.memcpy_htod(
                    self.input_buffers[self.input_names[2]]['buffer'], cond_cpu)

                # 设置绑定索引
                bindings = []
                for name in self.input_names:
                    bindings.append(int(self.input_buffers[name]['buffer']))
                for name in self.output_names:
                    bindings.append(int(self.output_buffers[name]['buffer']))

                # 执行推理
                self.context.execute_v2(bindings)

                # 获取输出
                out_name = self.output_names[0]
                out_buf = self.output_buffers[out_name]
                out_shape = out_buf['shape']
                out_dtype = out_buf['dtype']
                out_cpu = np.empty(trt.volume(out_shape), dtype=out_dtype)
                cuda.memcpy_dtoh(out_cpu, out_buf['buffer'])
                out_tensor = torch.from_numpy(
                    out_cpu).reshape(out_shape).to(x.device)

                # 返回格式与原始模型一致
                return out_tensor, None, None

            def eval(self):
                pass

            def half(self):
                return self

            def to(self, device):
                return self

        return TensorRTModel(engine, context, input_buffers, output_buffers, input_names, output_names)

    @torch.no_grad()
    def generate_batch(self, prompts: List[str], negative_prompts: List[str] = None,
                       duration: float = 2.0, fps: int = 8, cfg_scale: float = 7.5,
                       num_steps: int = 50, style=None, camera=None,
                       output_format="mp4", watermark=None, init_images=None,
                       reference_images=None, reference_videos=None, audio_paths=None,
                       lens_script_paths=None, prev_states=None, return_state=False,
                       output_paths=None, physics_trajectories=None, init_videos=None,
                       camera_trajs=None, resolution="256p",
                       return_latents=False) -> List[str]:
        """
        批量生成接口，主要用于 TileGenerator 并行处理
        新增 resolution 参数，用于动态计算潜变量尺寸
        新增 return_latents 参数，当为 True 时返回潜变量张量列表而非视频路径
        """
        batch_size = len(prompts)
        if negative_prompts is None:
            negative_prompts = [""] * batch_size

        # 编码所有文本（批量）
        text_embs = []
        neg_embs = []
        for p, np_ in zip(prompts, negative_prompts):
            text_emb, neg_emb = self._get_text_emb(p, np_)
            text_embs.append(text_emb)
            neg_embs.append(neg_emb)
        text_emb = torch.cat(text_embs, dim=0)
        neg_emb = torch.cat(neg_embs, dim=0)

        # 统一条件（仅文本）
        cond_emb = text_emb.mean(dim=1, keepdim=True)
        neg_emb = neg_emb.mean(dim=1, keepdim=True)
        if self.use_half:
            cond_emb = cond_emb.half()
            neg_emb = neg_emb.half()
        else:
            cond_emb = cond_emb.float()
            neg_emb = neg_emb.float()

        # 计算形状（所有视频相同）
        num_frames = int(duration * fps)
        target_size = self._resolution_to_pixel(resolution)
        h = target_size // self.vae.spatial_compress
        w = target_size // self.vae.spatial_compress
        t = num_frames // self.vae.temporal_compress
        if num_frames % self.vae.temporal_compress != 0:
            t += 1
        shape = (batch_size, self.vae.latent_channels, t, h, w)

        # 初始化 latents（支持 init_videos 或 init_images）
        if init_videos is not None and len(init_videos) == batch_size:
            latents_list = []
            for v in init_videos:
                z, _, _ = self.vae.encode(v)
                latents_list.append(z)
            latents = torch.cat(latents_list, dim=0)
            if latents.shape[2] != t:
                latents = F.interpolate(
                    latents, size=(t, h, w), mode='trilinear')
            noise = torch.randn_like(latents)
            t_step = torch.full(
                (batch_size,), self.scheduler.num_timesteps - 1, device=self.device)
            latents = self.scheduler.q_sample(latents, t_step, noise)
            if self.use_half:
                latents = latents.half()
            else:
                latents = latents.float()
        elif init_images is not None and len(init_images) == batch_size:
            img_tensors = []
            for img in init_images:
                if img is None:
                    img_tensors.append(None)
                    continue
                img = cv2.resize(img, (target_size, target_size))
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_t = torch.from_numpy(img).permute(
                    2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
                img_tensors.append(img_t.to(self.device))
            latents_list = []
            for img_t in img_tensors:
                if img_t is None:
                    latents_list.append(torch.zeros(
                        1, self.vae.latent_channels, 1, h, w, device=self.device))
                else:
                    z, _, _ = self.vae.encode(img_t)
                    latents_list.append(z)
            latents = torch.cat(latents_list, dim=0)
            latents = latents.repeat(1, 1, t, 1, 1)
            noise = torch.randn_like(latents)
            t_step = torch.full((batch_size,), 500, device=self.device)
            latents = self.scheduler.q_sample(latents, t_step, noise)
            if self.use_half:
                latents = latents.half()
            else:
                latents = latents.float()
        else:
            latents = torch.randn(shape, device=self.device)
            if self.use_half:
                latents = latents.half()
            else:
                latents = latents.float()

        self.scheduler.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.scheduler.timesteps

        # 扩散采样（批量）— 暂未集成 TeaCache（因批量场景不常用）
        for i, t_step in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t_step,
                                  device=self.device, dtype=torch.long)

            # 条件分支预测
            out_cond = self.model(latents, t_tensor, cond_emb)
            noise_pred = out_cond[0] if isinstance(
                out_cond, tuple) else out_cond
            # 无条件分支预测
            out_uncond = self.model(latents, t_tensor, neg_emb)
            noise_uncond = out_uncond[0] if isinstance(
                out_uncond, tuple) else out_uncond

            noise_pred = noise_uncond + cfg_scale * (noise_pred - noise_uncond)
            latents = self.scheduler.scheduler.step(
                noise_pred, t_step, latents).prev_sample

            # NaN 检测（批量）
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                raise RuntimeError(
                    f"NaN/Inf detected in batch generation at step {i}")

        # 如果需要返回潜变量，则直接返回
        if return_latents:
            return latents

        # 否则解码所有视频
        videos = self.vae.decode(latents).cpu().numpy()
        videos = (videos + 1) / 2 * 255
        videos = videos.transpose(0, 2, 3, 4, 1).astype(
            np.uint8)  # (B, T, H, W, C)
        videos = [v[:num_frames] for v in videos]

        # 后处理（批量）
        post_config = {'camera': camera, 'style': style}
        videos = [postprocess_video(
            v, post_config, mode=self.mode) for v in videos]
        if watermark:
            from postprocess import add_watermark
            videos = [add_watermark(v, watermark) for v in videos]

        # 保存视频
        if output_paths is None:
            output_paths = [
                f"output_{i}_{int(time.time())}.{output_format}" for i in range(batch_size)]
        for i, video in enumerate(videos):
            save_video(video, output_paths[i], fps)
        return output_paths

    def _resolution_to_pixel(self, resolution: str) -> int:
        """将分辨率字符串转换为像素边长（假设正方形）"""
        mapping = {
            "256p": 256, "360p": 360, "480p": 480, "720p": 720,
            "1080p": 1080, "1440p (2K)": 1440, "4K": 3840, "8K": 7680
        }
        if resolution.startswith("custom"):
            parts = resolution.split('_')[1].split('x')
            if len(parts) == 2:
                return max(int(parts[0]), int(parts[1]))
        return mapping.get(resolution, 256)

    def _get_text_emb(self, prompt, negative_prompt=""):
        key = (prompt, negative_prompt)
        if key in self.text_cache:
            return self.text_cache[key]
        text_emb = self.text_encoder([prompt])
        if negative_prompt:
            neg_emb = self.text_encoder([negative_prompt])
        else:
            neg_emb = torch.zeros_like(text_emb)
        if len(self.text_cache) >= self.max_cache_size:
            self.text_cache.pop(next(iter(self.text_cache)))
        self.text_cache[key] = (text_emb, neg_emb)
        return text_emb, neg_emb

    def build_cond_seq(self, text_emb, img_cond, audio_cond, video_cond, lens_cond, traj_cond):
        cond_list = []
        cond_list.append(text_emb.mean(dim=1, keepdim=True))
        if img_cond is not None:
            cond_list.append(img_cond)
        if audio_cond is not None:
            cond_list.append(audio_cond)
        if video_cond is not None:
            cond_list.append(video_cond)
        if lens_cond is not None:
            cond_list.append(lens_cond)
        if traj_cond is not None:
            cond_list.append(traj_cond)
        if len(cond_list) == 0:
            return None
        return torch.cat(cond_list, dim=1)

    # ========== 辅助方法 ==========
    def _tensor_to_frames(self, video_tensor):
        video_np = video_tensor.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        video_np = ((video_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return list(video_np)

    # ========== 主生成入口 ==========
    @torch.no_grad()
    def generate(self, prompt, negative_prompt="", duration=2.0, fps=8,
                 cfg_scale=7.5, num_steps=50, style=None, camera=None,
                 output_format="mp4", watermark=None, init_image=None,
                 reference_images=None, reference_videos=None, audio_paths=None,
                 lens_script_path=None, prev_state=None, return_state=False,
                 output_path=None, physics_trajectory=None, init_video=None,
                 camera_traj=None, resolution="256p",
                 lens_cond=None, boundary_mask=None, global_memory=None, **kwargs):
        """
        统一生成入口，根据时长自动选择生成路径。
        """
        if duration < 30:
            return self._generate_creative(
                prompt, negative_prompt, duration, fps,
                cfg_scale, num_steps, style, camera,
                output_format, watermark, init_image,
                reference_images, reference_videos, audio_paths,
                lens_script_path, output_path, init_video, resolution
            )
        else:
            # 从 kwargs 中提取 generate_long 需要的参数，避免重复传递
            use_pyramid = kwargs.pop('use_pyramid', False)
            use_parallel = kwargs.pop('use_parallel', False)
            num_gpus = kwargs.pop('num_gpus', 0)
            width = kwargs.pop('width', None)
            height = kwargs.pop('height', None)
            optimization_level = kwargs.pop(
                'optimization_level', 'full' if duration > 300 else 'light')
            lens_cond = kwargs.pop('lens_cond', None)
            boundary_mask = kwargs.pop('boundary_mask', None)
            global_memory = kwargs.pop('global_memory', None)

            return self.generate_long(
                prompt=prompt,
                duration=duration,
                fps=fps,
                resolution=resolution,
                width=width,
                height=height,
                cfg_scale=cfg_scale,
                steps=num_steps,
                negative_prompt=negative_prompt,
                prev_state=prev_state,
                output_path=output_path,
                lens_script_path=lens_script_path,
                use_parallel=use_parallel,
                num_gpus=num_gpus,
                use_pyramid=use_pyramid,
                style=style,
                camera=camera,
                watermark=watermark,
                init_image=init_image,
                reference_images=reference_images,
                reference_videos=reference_videos,
                audio_paths=audio_paths,
                physics_trajectory=physics_trajectory,
                init_video=init_video,
                camera_traj=camera_traj,
                lens_cond=lens_cond,
                boundary_mask=boundary_mask,
                global_memory=global_memory,
                optimization_level=optimization_level,
                **kwargs
            )

    def _generate_creative(self, prompt, negative_prompt, duration, fps,
                           cfg_scale, num_steps, style, camera,
                           output_format, watermark, init_image,
                           reference_images, reference_videos, audio_paths,
                           lens_script_path, output_path, init_video,
                           resolution="256p"):
        print(f"[创意模式] 生成视频，提示词：{prompt}")
        print(
            f"  参数：duration={duration}s, fps={fps}, num_steps={num_steps}, resolution={resolution}")
        sys.stdout.flush()
        num_frames = int(duration * fps)

        text_emb, neg_emb = self._get_text_emb(prompt, negative_prompt)
        cond_emb = text_emb.mean(dim=1, keepdim=True)
        neg_emb = neg_emb
        if self.use_half:
            cond_emb = cond_emb.half()
            neg_emb = neg_emb.half()
        else:
            cond_emb = cond_emb.float()
            neg_emb = neg_emb.float()

        if reference_images is not None and len(reference_images) > 0:
            img_tensors = []
            target_size = self._resolution_to_pixel(resolution)
            for img in reference_images[:self.config.model.max_reference_images]:
                img = cv2.resize(img, (target_size, target_size))
                img = torch.from_numpy(img).permute(
                    2, 0, 1).float() / 127.5 - 1.0
                img_tensors.append(img.unsqueeze(0).to(self.device))
            img_feats = self.image_encoder(img_tensors)
            img_cond = img_feats.mean(dim=1, keepdim=True)
            cond_emb = cond_emb + img_cond

        if reference_videos is not None and len(reference_videos) > 0:
            vid_list = []
            for vid in reference_videos[:self.config.model.max_reference_videos]:
                if isinstance(vid, np.ndarray):
                    vid = torch.from_numpy(vid).permute(
                        3, 0, 1, 2).float() / 127.5 - 1.0
                    vid = vid.unsqueeze(0)
                feat = self.video_encoder(vid)
                vid_list.append(feat)
            if vid_list:
                video_cond = torch.stack(
                    vid_list, dim=1).mean(dim=1, keepdim=True)
                cond_emb = cond_emb + video_cond

        if audio_paths is not None and len(audio_paths) > 0:
            audio_list = []
            for path in audio_paths[:self.config.model.max_reference_audios]:
                aud = self.audio_encoder.load_audio(path)
                feat = self.audio_encoder(aud)
                audio_list.append(feat.mean(dim=1, keepdim=True))
            if audio_list:
                audio_cond = torch.stack(audio_list, dim=1).mean(dim=1)
                cond_emb = cond_emb + audio_cond

        if lens_script_path is not None:
            shots = LensScriptParser.parse_script(lens_script_path)
            lens_cond = self.lens_controller(shots)
            lens_cond = lens_cond.mean(dim=1, keepdim=True)
            cond_emb = cond_emb + lens_cond

        target_size = self._resolution_to_pixel(resolution)
        h = target_size // self.vae.spatial_compress
        w = target_size // self.vae.spatial_compress
        t = num_frames // self.vae.temporal_compress
        if num_frames % self.vae.temporal_compress != 0:
            t += 1
        shape = (1, self.vae.latent_channels, t, h, w)

        if init_video is not None:
            latents = self.vae.encode(init_video)[0]
            if latents.shape[2] != t:
                latents = F.interpolate(
                    latents, size=(t, h, w), mode='trilinear')
            noise = torch.randn_like(latents)
            t_step = torch.full(
                (1,), self.scheduler.num_timesteps - 1, device=self.device)
            latents = self.scheduler.q_sample(latents, t_step, noise)
            if self.use_half:
                latents = latents.half()
            else:
                latents = latents.float()
        elif init_image is not None:
            init_image = cv2.resize(init_image, (target_size, target_size))
            if len(init_image.shape) == 2:
                init_image = cv2.cvtColor(init_image, cv2.COLOR_GRAY2RGB)
            elif init_image.shape[2] == 4:
                init_image = cv2.cvtColor(init_image, cv2.COLOR_BGRA2RGB)
            elif init_image.shape[2] == 3:
                init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB)
            init_image = torch.from_numpy(init_image).permute(
                2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            init_image = init_image.to(self.device)
            latents = self.vae.encode(init_image)[0]
            latents = latents.repeat(1, 1, t, 1, 1)
            noise = torch.randn_like(latents)
            t_step = torch.full((1,), 500, device=self.device)
            latents = self.scheduler.q_sample(latents, t_step, noise)
            if self.use_half:
                latents = latents.half()
            else:
                latents = latents.float()
        else:
            latents = torch.randn(shape, device=self.device)
            if self.use_half:
                latents = latents.half()
            else:
                latents = latents.float()

        self.scheduler.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.scheduler.timesteps

        if self.tea_cache is not None:
            self.tea_cache.reset()
        print(f"  扩散步数总数：{len(timesteps)}")
        for i, t_step in enumerate(timesteps):
            print(f"  步数 {i + 1}/{len(timesteps)} (t={t_step})", flush=True)
            t_tensor = torch.full(
                (1,), t_step, device=self.device, dtype=torch.long)

            def model_forward(latents, t, cond):
                output = self.model(latents, t, cond)
                return output[0] if isinstance(output, tuple) else output

            if self.tea_cache is not None:
                noise_pred, used_cache = self.tea_cache.step(
                    latents, t_tensor, cond_emb, model_forward)
            else:
                noise_pred = model_forward(latents, t_tensor, cond_emb)

            # 无条件分支预测
            out_uncond = self.model(latents, t_tensor, neg_emb)
            noise_uncond = out_uncond[0] if isinstance(
                out_uncond, tuple) else out_uncond

            noise_pred = noise_uncond + cfg_scale * (noise_pred - noise_uncond)
            latents = self.scheduler.scheduler.step(
                noise_pred, t_step, latents).prev_sample

            # ========== 数值稳定性：检测 NaN/Inf 并限制范围 ==========
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                print(f"❌ 在第 {i+1} 步检测到 NaN/Inf！")
                raise RuntimeError("NaN/Inf in latents")
            # 可选：限制 latent 范围，防止后续爆炸
            latents = torch.clamp(latents, -3.0, 3.0)
        print("开始 VAE 解码...")
        video = self.vae.decode(latents).cpu().numpy()
        video = (video + 1) / 2 * 255
        video = video.squeeze(0).transpose(
            1, 2, 3, 0).astype(np.uint8)[:num_frames]
        print("VAE 解码完成")
        post_config = {'camera': camera, 'style': style}
        video = postprocess_video(video, post_config, mode='creative')
        if watermark:
            from postprocess import add_watermark
            video = add_watermark(video, watermark)

        if output_path is None:
            output_path = f"output_creative_{int(time.time())}.{output_format}"
        save_video(video, output_path, fps)
        return output_path

    def _generate_long(self, prompt, negative_prompt, duration, fps,
                       cfg_scale, num_steps, style, camera,
                       output_format, watermark, init_image,
                       reference_images, reference_videos, audio_paths,
                       lens_script_path, prev_state, return_state,
                       output_path, physics_trajectory, init_video, camera_traj,
                       resolution="256p", **kwargs):
        kwargs.pop('use_pyramid', None)
        kwargs.pop('use_parallel', None)
        kwargs.pop('num_gpus', None)
        kwargs.pop('width', None)
        kwargs.pop('height', None)
        optimization_level = kwargs.pop('optimization_level', 'light')
        lens_cond = kwargs.pop('lens_cond', None)
        boundary_mask = kwargs.pop('boundary_mask', None)
        global_memory = kwargs.pop('global_memory', None)

        return self.generate_long(
            prompt=prompt,
            duration=duration,
            fps=fps,
            resolution=resolution,
            width=None,
            height=None,
            cfg_scale=cfg_scale,
            steps=num_steps,
            negative_prompt=negative_prompt,
            prev_state=prev_state,
            output_path=output_path,
            lens_script_path=lens_script_path,
            use_parallel=False,
            num_gpus=0,
            use_pyramid=False,
            style=style,
            camera=camera,
            watermark=watermark,
            init_image=init_image,
            reference_images=reference_images,
            reference_videos=reference_videos,
            audio_paths=audio_paths,
            physics_trajectory=physics_trajectory,
            init_video=init_video,
            camera_traj=camera_traj,
            lens_cond=lens_cond,
            boundary_mask=boundary_mask,
            global_memory=global_memory,
            optimization_level=optimization_level,
            **kwargs
        )

    # ========== 新增方法：生成单个块的潜变量 ==========
    def _generate_block_latents(self, prompt, num_frames, fps, cfg_scale, steps,
                                negative_prompt, lens_cond, boundary_mask,
                                global_memory, resolution, **kwargs):
        """
        生成一个块的潜变量（不进行 VAE 解码）
        假设 tile_generator 支持 return_latents=True 参数
        """
        # 根据分辨率调用相应的 tile_generator 方法，并传入 return_latents=True
        # 注意：需要修改 tile_generator 的 generate_1080p 等方法以支持该参数
        if resolution == "1080p":
            latents = self.tile_generator.generate_1080p(
                prompt, num_frames, fps, cfg_scale=cfg_scale, num_steps=steps,
                negative_prompt=negative_prompt, prev_state=None,
                lens_script_path=None,
                lens_cond=lens_cond, boundary_mask=boundary_mask,
                global_memory=global_memory, return_latents=True, **kwargs)
        elif resolution == "4k":
            latents = self.tile_generator.generate_4k(
                prompt, num_frames, fps, cfg_scale=cfg_scale, num_steps=steps,
                negative_prompt=negative_prompt, prev_state=None,
                lens_script_path=None,
                lens_cond=lens_cond, boundary_mask=boundary_mask,
                global_memory=global_memory, return_latents=True, **kwargs)
        elif resolution == "8k":
            latents = self.tile_generator.generate_8k(
                prompt, num_frames, fps, cfg_scale=cfg_scale, num_steps=steps,
                negative_prompt=negative_prompt, prev_state=None,
                lens_script_path=None,
                lens_cond=lens_cond, boundary_mask=boundary_mask,
                global_memory=global_memory, return_latents=True, **kwargs)
        else:
            raise ValueError(
                f"Unsupported resolution for block generation: {resolution}")
        return latents

    def generate_long(self, prompt, duration=30, fps=24, resolution="1080p",
                      width=None, height=None, cfg_scale=7.5, steps=50,
                      negative_prompt="", prev_state=None, output_path=None,
                      lens_script_path=None, use_parallel=False, num_gpus=0,
                      use_pyramid=False, optimization_level='light',
                      lens_cond=None, boundary_mask=None, global_memory=None, **kwargs):
        # 从 kwargs 中提取 use_pipeline 标志，默认 False
        use_pipeline = kwargs.pop('use_pipeline', False)

        # 根据优化等级调整配置
        if optimization_level == 'light':
            self.config.model.use_internal_memory = False
            self.config.model.use_multi_scale_memory = False
            self.config.model.use_adaptive_compressor = False
            self.config.model.use_memory_bank = False
            self.config.model.use_layerwise_history = False
            self.config.model.use_phy_experts = False
            use_parallel = False
            use_pyramid = False
        else:
            self.config.model.use_internal_memory = True
            self.config.model.use_multi_scale_memory = True
            self.config.model.use_adaptive_compressor = True
            self.config.model.use_memory_bank = True
            self.config.model.use_layerwise_history = True
            self.config.model.use_phy_experts = True
            if num_gpus <= 0:
                num_gpus = torch.cuda.device_count()
            use_parallel = (num_gpus > 1)
            use_pyramid = True

        if lens_cond is None and lens_script_path is not None:
            shots = LensScriptParser.parse_script(lens_script_path)
            lens_cond, boundary_mask = self.lens_controller(shots, fps=fps)

        # 金字塔采样（取消分辨率限制）
        # MODIFIED: 去掉分辨率限制，对所有分辨率允许金字塔采样
        if use_pyramid:
            low_res = getattr(self.config.model,
                              'pyramid_base_resolution', "128p")
            low_steps = getattr(self.config.model, 'pyramid_base_steps', 10)
            print(f"使用金字塔采样：先以 {low_res} 生成 {low_steps} 步，再细化到 {resolution}")
            low_path, low_state = self.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                duration=duration,
                fps=fps,
                num_steps=low_steps,
                cfg_scale=cfg_scale,
                output_path=None,
                return_state=True,
                **kwargs
            )
            low_video = self._load_video_tensor(low_path)
            target_w, target_h = self._resolution_to_wh(resolution)
            upsampled = F.interpolate(low_video, size=(
                target_h, target_w), mode='bilinear', align_corners=False)

            superres_model = getattr(
                self.config.model, 'superres_model', 'RealESRGAN_x4plus')
            superres_workers = getattr(
                self.config.model, 'superres_parallel_workers', 4)
            refine_steps = getattr(self.config.model, 'refine_steps', 0)

            if refine_steps > 0:
                print(f"使用扩散精炼：{refine_steps} 步")
                output_path, _ = self.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    duration=duration,
                    fps=fps,
                    num_steps=refine_steps,
                    cfg_scale=cfg_scale,
                    init_video=upsampled,
                    output_path=output_path,
                    **kwargs
                )
            else:
                print(f"使用超分模型：{superres_model}")
                from postprocess.superres import upscale_video
                frames = self._tensor_to_frames(upsampled)
                upscaled_frames = upscale_video(
                    frames,
                    scale=1,
                    model_name=superres_model,
                    num_workers=superres_workers
                )
                output_path = output_path or os.path.join(
                    self.config.api.temp_dir, f"output_{int(time.time())}.mp4")
                save_video(upscaled_frames, output_path, fps)
            return output_path, None

        if use_parallel and num_gpus > 1:
            return self.generate_parallel(prompt, duration, fps, resolution, num_gpus, cfg_scale, steps,
                                          negative_prompt, lens_script_path, **kwargs)

        if width is not None and height is not None:
            target_w, target_h = width, height
        else:
            target_map = {"360p": (640, 360), "480p": (854, 480), "720p": (1280, 720),
                          "1080p": (1920, 1080), "1440p": (2560, 1440), "4k": (3840, 2160),
                          "8k": (7680, 4320), "16k": (15360, 8640), "256p": (256, 256)}
            target_w, target_h = target_map.get(resolution, (256, 256))

        if resolution == "256p":
            style = kwargs.get('style')
            camera = kwargs.get('camera')
            watermark = kwargs.get('watermark')
            init_image = kwargs.get('init_image')
            reference_images = kwargs.get('reference_images')
            reference_videos = kwargs.get('reference_videos')
            audio_paths = kwargs.get('audio_paths')
            init_video = kwargs.get('init_video')
            out_path = self._generate_creative(
                prompt=prompt,
                negative_prompt=negative_prompt,
                duration=duration,
                fps=fps,
                cfg_scale=cfg_scale,
                num_steps=steps,
                style=style,
                camera=camera,
                output_format='mp4',
                watermark=watermark,
                init_image=init_image,
                reference_images=reference_images,
                reference_videos=reference_videos,
                audio_paths=audio_paths,
                lens_script_path=lens_script_path,
                output_path=output_path,
                init_video=init_video,
                resolution=resolution
            )
            return out_path, None

        camera_traj = None
        if lens_script_path is not None:
            try:
                with open(lens_script_path, 'r') as f:
                    script = json.load(f)
                if 'camera_trajectory' in script:
                    traj_list = script['camera_trajectory']
                    traj_tensor = torch.tensor(
                        traj_list, dtype=torch.float32, device=self.device).unsqueeze(0)
                    camera_traj = self.traj_proj(
                        traj_tensor).mean(dim=1, keepdim=True)
            except Exception as e:
                print(f"警告：解析摄像机轨迹失败：{e}")

        max_block_frames = getattr(self.config.model, 'max_block_frames', 192)
        block_duration = max_block_frames / fps
        num_blocks = int(np.ceil(duration / block_duration))

        block_videos = []
        state = prev_state

        if use_pipeline:
            # 流水线并行：后处理与下一块生成重叠（异步解码）
            executor = ThreadPoolExecutor(max_workers=2)
            gen_futures = []   # 存储生成潜变量的 Future
            decode_futures = []  # 存储解码结果的 Future

            for i in range(num_blocks):
                block_dur = block_duration if i < num_blocks - \
                    1 else duration - i * block_duration
                if block_dur <= 0:
                    break
                block_frames = int(block_dur * fps)

                block_boundary = None
                if boundary_mask is not None:
                    block_boundary = boundary_mask
                block_lens_cond = lens_cond

                if self.tea_cache is not None:
                    self.tea_cache.reset()

                print(f"生成块 {i+1}/{num_blocks} 的潜变量")

                # 提交生成潜变量的任务
                future = executor.submit(
                    self._generate_block_latents,
                    prompt, block_frames, fps, cfg_scale, steps,
                    negative_prompt, block_lens_cond, block_boundary,
                    global_memory, resolution, **kwargs
                )
                gen_futures.append(future)

                # 如果上一个块已经生成完成，则提交解码任务
                if i > 0:
                    prev_future = gen_futures[i-1]
                    # 等待上一个块的潜变量结果（非阻塞，因为 submit 后直接继续）
                    prev_latents = prev_future.result()  # 这里会阻塞直到完成，但我们可以用回调方式
                    # 提交解码任务
                    decode_future = executor.submit(
                        self._decode_latents_to_video, prev_latents, fps, self.config, self.vae
                    )
                    decode_futures.append((i-1, decode_future))

                state = None

            # 处理最后一个块的解码
            last_future = gen_futures[-1]
            last_latents = last_future.result()
            decode_future = executor.submit(
                self._decode_latents_to_video, last_latents, fps, self.config, self.vae
            )
            decode_futures.append((num_blocks-1, decode_future))

            # 收集解码结果，按块顺序
            block_videos = [None] * num_blocks
            for idx, fut in decode_futures:
                block_videos[idx] = fut.result()

            executor.shutdown(wait=True)

        else:
            # 非流水线模式：顺序生成
            for i in range(num_blocks):
                block_dur = block_duration if i < num_blocks - \
                    1 else duration - i * block_duration
                if block_dur <= 0:
                    break
                block_frames = int(block_dur * fps)

                block_boundary = None
                if boundary_mask is not None:
                    block_boundary = boundary_mask
                block_lens_cond = lens_cond

                if self.tea_cache is not None:
                    self.tea_cache.reset()

                if resolution == "1080p":
                    highres_video = self.tile_generator.generate_1080p(
                        prompt, block_frames, fps, cfg_scale=cfg_scale, num_steps=steps,
                        negative_prompt=negative_prompt, prev_state=state,
                        lens_script_path=lens_script_path,
                        lens_cond=block_lens_cond, boundary_mask=block_boundary,
                        global_memory=global_memory, **kwargs)
                elif resolution == "4k":
                    highres_video = self.tile_generator.generate_4k(
                        prompt, block_frames, fps, cfg_scale=cfg_scale, num_steps=steps,
                        negative_prompt=negative_prompt, prev_state=state,
                        lens_script_path=lens_script_path,
                        lens_cond=block_lens_cond, boundary_mask=block_boundary,
                        global_memory=global_memory, **kwargs)
                elif resolution == "8k":
                    highres_video = self.tile_generator.generate_8k(
                        prompt, block_frames, fps, cfg_scale=cfg_scale, num_steps=steps,
                        negative_prompt=negative_prompt, prev_state=state,
                        lens_script_path=lens_script_path,
                        lens_cond=block_lens_cond, boundary_mask=block_boundary,
                        global_memory=global_memory, **kwargs)
                else:
                    raise ValueError(f"Unsupported resolution: {resolution}")

                if self.config.model.use_physics_correction:
                    from postprocess.physics_corrector import correct_video
                    highres_video = correct_video(
                        highres_video, fps, self.config)

                block_videos.append(highres_video)
                state = None

        full_video = np.concatenate(block_videos, axis=0) if len(
            block_videos) > 1 else block_videos[0]

        if not use_pipeline and self.config.model.use_physics_correction:
            print(
                f"[Physics] Applying light correction (Solver: {self.config.model.phy_corr_solver}, Interval: {self.config.model.phy_corr_keyframe_interval})")
            from postprocess.physics_corrector import correct_video
            full_video = correct_video(full_video, fps, self.config)

        if output_path is None:
            output_path = os.path.join(
                self.config.api.temp_dir, f"output_{int(time.time())}.mp4")
        save_video(full_video, output_path, fps)
        return output_path, None

    # ========== 新增辅助方法：潜变量解码 ==========
    def _decode_latents_to_video(self, latents, fps, config, vae):
        """
        将潜变量解码为视频数组
        """
        video = vae.decode(latents).cpu().numpy()
        video = (video + 1) / 2 * 255
        video = video.squeeze(0).transpose(1, 2, 3, 0).astype(np.uint8)
        # 后处理（物理校正等）
        if config.model.use_physics_correction:
            from postprocess.physics_corrector import correct_video
            video = correct_video(video, fps, config)
        return video

    # ========== 新增流水线并行生成方法 ==========
    def generate_pipeline(self, prompt, duration=30, fps=24, resolution="1080p",
                          cfg_scale=7.5, steps=50, negative_prompt="", prev_state=None,
                          output_path=None, lens_script_path=None, **kwargs):
        """
        流水线并行生成视频：将生成与后处理（含物理校正）重叠，提升长视频生成效率。
        此方法将调用 generate_long 并启用 use_pipeline 标志。
        """
        return self.generate_long(
            prompt=prompt,
            duration=duration,
            fps=fps,
            resolution=resolution,
            cfg_scale=cfg_scale,
            steps=steps,
            negative_prompt=negative_prompt,
            prev_state=prev_state,
            output_path=output_path,
            lens_script_path=lens_script_path,
            use_pipeline=True,
            **kwargs
        )

    # ========== 其他现有方法（保持不变） ==========
    def generate_parallel(self, prompt, duration, fps=24, resolution="720p",
                          num_gpus=1, overlap_sec=0.5, cfg_scale=7.5, steps=50,
                          negative_prompt="", lens_script_path=None, **kwargs):
        """
        多 GPU 并行生成（基于时间分块）。
        将视频按时间切分成多个块，分配给各 GPU 独立生成，最后融合。
        每个 GPU 独立加载模型，显存占用高，但速度接近线性加速。

        参数:
            num_gpus: 使用的 GPU 数量，默认 1（自动检测可用 GPU）
            overlap_sec: 块间重叠时长（秒），用于平滑拼接
        """
        import torch.multiprocessing as mp
        import numpy as np
        from utils import save_video
        import time
        import os

        # 确定实际使用的 GPU 数量
        available_gpus = torch.cuda.device_count()
        if num_gpus > available_gpus:
            print(
                f"警告：指定的 GPU 数量 {num_gpus} 超过可用数量 {available_gpus}，将使用 {available_gpus} 个")
            num_gpus = available_gpus
        if num_gpus <= 1:
            return self.generate_long(prompt, duration, fps, resolution,
                                      cfg_scale=cfg_scale, steps=steps,
                                      negative_prompt=negative_prompt,
                                      lens_script_path=lens_script_path, **kwargs)

        # 计算时间块
        total_frames = int(duration * fps)
        overlap_frames = int(overlap_sec * fps)
        # 按块数分配，使每块长度尽量相等
        segment_frames = total_frames // num_gpus
        gpu_segments = []
        for gpu_id in range(num_gpus):
            start_frame = gpu_id * segment_frames
            end_frame = start_frame + segment_frames
            # 添加重叠（非首尾块）
            if gpu_id > 0:
                start_frame = max(0, start_frame - overlap_frames)
            if gpu_id < num_gpus - 1:
                end_frame = min(total_frames, end_frame + overlap_frames)
            gpu_segments.append((start_frame, end_frame))

        # 定义工作函数
        def worker(gpu_id, start_frame, end_frame, result_queue):
            try:
                torch.cuda.set_device(gpu_id)
                device = torch.device(f'cuda:{gpu_id}')

                # 重新加载模型（每个进程独立）
                from config import Config
                from models.vae import VideoVAE
                from models.dit import SpatialTemporalUNet
                from models.text_encoder import TextEncoder
                from models.diffusion import DiffusionScheduler
                from models.lens_controller import LensController
                from inferencer import Inferencer

                config = self.config
                model = SpatialTemporalUNet(config.model).to(device).half()
                vae = VideoVAE(config.model, device=device)
                scheduler = DiffusionScheduler(config.model, device)
                text_encoder = TextEncoder(
                    config.model.text_encoder, config.model.text_encoder_model, device)
                lens_controller = LensController(config.model)

                infer = Inferencer(config, model, vae, scheduler, text_encoder,
                                   None, None, None, lens_controller, device)

                # 生成该段的视频
                seg_duration = (end_frame - start_frame) / fps
                seg_prompt = prompt
                # 可传入镜头脚本等
                seg_kwargs = kwargs.copy()
                seg_kwargs['lens_script_path'] = lens_script_path
                output_path, _ = infer.generate_long(
                    prompt=seg_prompt,
                    duration=seg_duration,
                    fps=fps,
                    resolution=resolution,
                    cfg_scale=cfg_scale,
                    steps=steps,
                    negative_prompt=negative_prompt,
                    output_path=None,
                    **seg_kwargs
                )
                # 读取生成的视频
                cap = cv2.VideoCapture(output_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                video = np.array(frames)
                result_queue.put((gpu_id, start_frame, end_frame, video))
                # 清理临时文件
                try:
                    os.remove(output_path)
                except:
                    pass
            except Exception as e:
                print(f"GPU {gpu_id} 生成失败: {e}")
                result_queue.put((gpu_id, start_frame, end_frame, None))

        # 启动进程
        mp.set_start_method('spawn', force=True)
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        processes = []
        for gpu_id, (start, end) in enumerate(gpu_segments):
            p = ctx.Process(target=worker, args=(
                gpu_id, start, end, result_queue))
            p.start()
            processes.append(p)

        # 收集结果
        results = []
        for _ in processes:
            gpu_id, start, end, video = result_queue.get()
            if video is not None:
                results.append((start, end, video))
            else:
                print(f"GPU {gpu_id} 生成失败，跳过该段")

        # 等待所有进程结束
        for p in processes:
            p.join()

        # 按起始帧排序，合并视频
        results.sort(key=lambda x: x[0])  # 按 start 排序
        full_video = []
        for i, (start, end, video) in enumerate(results):
            if i == 0:
                full_video = list(video)
            else:
                overlap = overlap_frames
                if overlap > 0:
                    # 融合重叠部分
                    alpha = np.linspace(0, 1, overlap).reshape(-1, 1, 1, 1)
                    prev_overlap = full_video[-overlap:]
                    curr_overlap = video[:overlap]
                    blended = prev_overlap * (1 - alpha) + curr_overlap * alpha
                    full_video[-overlap:] = blended
                    full_video.extend(video[overlap:])
                else:
                    full_video.extend(video)
        full_video = np.array(full_video)[:total_frames]

        # 保存最终视频
        output_path = f"output_parallel_{int(time.time())}.mp4"
        save_video(full_video, output_path, fps)
        return output_path

    def generate_4k(self, prompt, duration=10.0, fps=24, **kwargs):
        num_frames = int(duration * fps)
        return self.tile_generator.generate_4k(prompt, num_frames, fps, **kwargs)

    def generate_8k(self, prompt, duration=10.0, fps=24, **kwargs):
        num_frames = int(duration * fps)
        return self.tile_generator.generate_8k(prompt, num_frames, fps, **kwargs)

    def generate_ar(self, prompt, duration=10.0, fps=24, cfg_scale=7.5, steps=50,
                    negative_prompt="", use_memory=False, output_path=None,
                    lens_script_path=None, keyframes=None, physics_trajectory=None):
        num_frames = int(duration * fps)
        state = self.model.get_initial_state(batch_size=1)
        memory_bank = MemoryBank(
            dim=self.config.model.dit_context_dim) if use_memory else None

        cond = self.text_encoder([prompt])
        if negative_prompt:
            neg_emb = self.text_encoder([negative_prompt])
        else:
            neg_emb = torch.zeros_like(cond)

        lens_cond = None
        if lens_script_path is not None:
            shots = LensScriptParser.parse_script(lens_script_path)
            lens_cond = self.lens_controller(shots)
            lens_cond = lens_cond.mean(dim=1, keepdim=True)

        traj_cond = None
        if physics_trajectory is not None:
            traj_tensor = torch.tensor(
                physics_trajectory, dtype=torch.float32, device=self.device)
            traj_cond = self.traj_proj(traj_tensor).mean(dim=0, keepdim=True)

        cond_emb = cond.mean(dim=1, keepdim=True)
        if lens_cond is not None:
            cond_emb = cond_emb + lens_cond
        if traj_cond is not None:
            cond_emb = cond_emb + traj_cond
        if self.use_half:
            cond_emb = cond_emb.half()
            neg_emb = neg_emb.half()
        else:
            cond_emb = cond_emb.float()
            neg_emb = neg_emb.float()

        init_latents = None
        if keyframes is not None and len(keyframes) > 1:
            interpolator = RIFEInterpolator(device=self.device)
            init_frames = self._interpolate_keyframes(
                keyframes, num_frames, interpolator)
            init_latents = self._frames_to_latents(init_frames)

        h = self.config.data.image_size // self.vae.spatial_compress
        w = self.config.data.image_size // self.vae.spatial_compress
        t = 1
        shape = (1, self.vae.latent_channels, t, h, w)

        if init_latents is not None:
            latents = init_latents[:, :, 0:1, :, :]
        else:
            latents = torch.randn(shape, device=self.device)
        if self.use_half:
            latents = latents.half()
        else:
            latents = latents.float()

        all_latents = []
        for frame_idx in range(num_frames):
            self.scheduler.scheduler.set_timesteps(steps)
            timesteps = self.scheduler.scheduler.timesteps

            if init_latents is not None and frame_idx < init_latents.shape[2]:
                init_frame = init_latents[:, :, frame_idx:frame_idx + 1, :, :]
                latents = 0.5 * latents + 0.5 * init_frame

            for i, t_step in enumerate(timesteps):
                print(f"  步数 {i + 1}/{len(timesteps)} (t={t_step})")
                t_tensor = torch.full(
                    (1,), t_step, device=self.device, dtype=torch.long)

                memory = None
                if use_memory and memory_bank is not None and frame_idx > 0:
                    memory = memory_bank.retrieve(latents)

                # 条件分支预测
                out_cond = self.model(latents, t_tensor, cond_emb,
                                      prev_state=state, memory=memory, use_cache=(frame_idx > 0))
                noise_pred = out_cond[0] if isinstance(
                    out_cond, tuple) else out_cond
                # 无条件分支预测
                out_uncond = self.model(latents, t_tensor, neg_emb,
                                        prev_state=state, use_cache=True)
                noise_uncond = out_uncond[0] if isinstance(
                    out_uncond, tuple) else out_uncond

                noise_pred = noise_uncond + cfg_scale * \
                    (noise_pred - noise_uncond)
                latents = self.scheduler.scheduler.step(
                    noise_pred, t_step, latents).prev_sample

                # NaN 检测
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    raise RuntimeError(f"NaN/Inf at step {i} in generate_ar")

            all_latents.append(latents)

            if use_memory and memory_bank is not None:
                B, C, T, H, W = latents.shape
                token_seq = latents.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
                feat = self.model.input_proj(
                    token_seq).mean(dim=1, keepdim=True)
                memory_bank.update(feat)

            self.model.reset_cache()

        full_latents = torch.cat(all_latents, dim=2)
        video = self.vae.decode(full_latents).cpu().numpy()
        video = (video + 1) / 2 * 255
        video = video.squeeze(0).transpose(
            1, 2, 3, 0).astype(np.uint8)[:num_frames]

        if output_path is None:
            output_path = f"output_ar_{int(time.time())}.mp4"
        save_video(video, output_path, fps)
        return output_path

    def generate_two_stage(self, prompt, duration, fps=24,
                           target_resolution="720p",
                           rough_resolution="128p", rough_fps=8, rough_steps=10,
                           refine_steps=20, use_memory=True, output_path=None,
                           lens_script_path=None):
        rough_path, _ = self.generate_long(
            prompt=prompt, duration=duration, fps=rough_fps,
            resolution=rough_resolution, steps=rough_steps,
            cfg_scale=7.5, negative_prompt="", lens_script_path=lens_script_path,
            return_state=True, output_path=None)
        rough_video = self._load_video_tensor(rough_path)

        target_w, target_h = self._resolution_to_wh(target_resolution)
        target_frames = int(duration * fps)
        T_rough = rough_video.shape[2]
        block_size = self.config.model.max_block_frames

        memory_bank = MemoryBank(
            dim=self.config.model.dit_context_dim) if use_memory else None

        refined_frames = []
        for i in range(T_rough - 1):
            start = i * (target_frames // T_rough)
            end = (i + 1) * (target_frames // T_rough)
            block_length = end - start
            if block_length <= 0:
                continue

            rough_frame_curr = rough_video[:, :, i, :, :]
            init_frame = self._upscale_frame(
                rough_frame_curr, target_w, target_h)
            init_latent = self._frame_to_latent(init_frame)

            refined_block = self._generate_block_ar(
                prompt=prompt, num_frames=block_length, fps=fps,
                cfg_scale=7.5, steps=refine_steps,
                init_latent=init_latent, memory_bank=memory_bank,
                lens_script_path=lens_script_path)
            refined_frames.append(refined_block)

            if use_memory:
                feat = self._extract_feature(rough_frame_curr)
                memory_bank.update(feat)

        final_video = np.concatenate(refined_frames, axis=0)
        if output_path is None:
            output_path = f"output_two_stage_{int(time.time())}.mp4"
        save_video(final_video, output_path, fps)
        return output_path

    # ========== 辅助方法（现有） ==========
    def _interpolate_keyframes(self, keyframes, target_frames, interpolator):
        import numpy as np
        keyframes_np = [k.astype(np.float32) / 255.0 for k in keyframes]
        frames = []
        for i in range(len(keyframes_np) - 1):
            start = keyframes_np[i]
            end = keyframes_np[i + 1]
            steps = target_frames // (len(keyframes_np) - 1)
            for j in range(steps):
                t = j / steps
                interp = interpolator.interpolate(start, end, t)
                frames.append(interp)
        while len(frames) < target_frames:
            frames.append(frames[-1])
        return np.array(frames[:target_frames])

    def _frames_to_latents(self, frames):
        frames_tensor = torch.from_numpy(frames).permute(
            0, 3, 1, 2).float().to(self.device)
        frames_tensor = (frames_tensor / 127.5) - 1.0
        with torch.no_grad():
            latents, _, _ = self.vae.encode(frames_tensor.unsqueeze(0))
        return latents

    def _load_video_tensor(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                frame, (self.config.data.image_size, self.config.data.image_size))
            frames.append(frame)
        cap.release()
        video = np.array(frames, dtype=np.float32) / 127.5 - 1.0
        video = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0)
        return video.to(self.device)

    def _resolution_to_wh(self, resolution):
        mapping = {"256p": (256, 256), "360p": (640, 360), "480p": (854, 480),
                   "720p": (1280, 720), "1080p": (1920, 1080), "1440p (2K)": (2560, 1440),
                   "4K": (3840, 2160), "8K": (7680, 4320)}
        return mapping.get(resolution, (256, 256))

    def _upscale_frame(self, frame_tensor, target_w, target_h):
        return F.interpolate(frame_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

    def _frame_to_latent(self, frame_tensor):
        with torch.no_grad():
            latent, _, _ = self.vae.encode(frame_tensor)
        return latent

    def _generate_block_ar(self, prompt, num_frames, fps, cfg_scale, steps,
                           init_latent, memory_bank, lens_script_path):
        raise NotImplementedError("Block AR generation not yet implemented")

    def _extract_feature(self, frame_tensor):
        B, C, H, W = frame_tensor.shape
        token_seq = frame_tensor.reshape(B, -1, C)
        feat = self.model.input_proj(token_seq).mean(dim=1, keepdim=True)
        return feat

    def _simulate_physics(self, object_type, params, duration, fps):
        import numpy as np
        if object_type == 'ball':
            v0 = params.get('velocity', [10.0, 10.0, 0.0])
            g = 9.8
            t = np.linspace(0, duration, int(duration * fps))
            x = v0[0] * t
            y = v0[1] * t
            z = v0[2] * t - 0.5 * g * t ** 2
            z = np.maximum(z, 0)
            trajectory = np.stack([x, y, z], axis=1)
            return trajectory
        else:
            raise ValueError(f"Unknown object type: {object_type}")

    def generate_pyramid(self, prompt, duration, fps, target_resolution="720p",
                         base_resolution="128p", base_steps=10, refine_steps=20,
                         **kwargs):
        base_path = self.generate(
            prompt, duration=duration, fps=fps,
            num_steps=base_steps,
            output_path=None, **kwargs
        )
        base_tensor = self._load_video_tensor(base_path)
        target_w, target_h = self._resolution_to_wh(target_resolution)
        upsampled = F.interpolate(base_tensor, size=(
            target_h, target_w), mode='bilinear', align_corners=False)
        refined_path = self.generate(
            prompt, duration=duration, fps=fps,
            num_steps=refine_steps,
            init_video=upsampled, **kwargs
        )
        return refined_path
