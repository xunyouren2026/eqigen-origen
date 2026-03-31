"""
深度调试版：定位模型 forward 中崩溃的具体位置
修改点：兼容模型返回 2 个值（默认）或 3 个值（return_state=True）
"""

import torch
import sys
import traceback
import faulthandler
import os

# 启用段错误跟踪
faulthandler.enable()

# 强制使用 CPU（避免 CUDA 相关崩溃）
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_default_device('cpu')

sys.path.insert(0, r'E:\video_gen')

try:
    from models.dit import SpatialTemporalUNet

    # 补全配置（与之前相同，确保所有属性都存在）
    class DummyConfig:
        dit_context_dim = 768
        dit_depth = 12
        dit_num_heads = 12
        dit_mlp_ratio = 4.0
        vae_latent_channels = 4
        attn_type = 'sliding_window'
        window_size = 512
        block_size = 64
        stride = 4
        cache_window_size = 512
        use_relative_position = False
        use_token_routing = False
        num_routing_blocks = 8
        routing_top_k = 4
        use_hierarchical_compression = False
        compress_ratios = (1, 4, 16)
        use_learned_compressor = False
        compressor_type = 'mlp'
        compressed_tokens = 64
        compressor_layers = 2
        use_jitter = False
        jitter_scale = 1e-5
        calib_mask_path = ''
        calib_top_k = 16
        use_memory = False
        use_phy_experts = False
        num_phy_experts = 4
        use_internal_memory = False
        use_memory_bank = False
        use_multi_scale_memory = False
        use_hierarchical_memory = False
        use_adaptive_compressor = False
        use_layerwise_history = False
        use_learned_retrieval = False
        use_importance_pred = False
        use_turboquant = False
        multi_modal_fusion = 'sum'
        use_unified_cond = False
        use_image_encoder = False
        use_audio_encoder = False
        use_video_encoder = False
        use_first_frame_anchor = False
        use_concatenated_history = False
        use_frame_corruption = False
        corruption_prob = 0.15
        corruption_num_modes = 2
        use_rope_jitter = False
        rope_jitter_scale = 1e-5
        use_packforcing = False
        motion_adaptive = False
        memory_size = 256
        memory_slots = 64
        memory_rank = 16
        memory_top_k = 8
        memory_controller = 'gru'
        use_linear_fusion = True

    config = DummyConfig()
    print("初始化模型...")
    model = SpatialTemporalUNet(config)
    model.eval()
    print("模型初始化成功。")

    # 打印参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")

    # 准备输入
    device = torch.device('cpu')
    model = model.to(device)

    B, C, T, H, W = 1, 4, 8, 32, 32
    x = torch.randn(B, C, T, H, W, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    cond = torch.randn(B, 1, config.dit_context_dim, device=device)

    print("开始前向传播...")

    # 逐步打印每个 block 的执行情况
    def forward_hook(module, input, output):
        print(f"  {module.__class__.__name__} forward done")

    # 为每个 block 注册 hook
    for i, block in enumerate(model.blocks):
        block.register_forward_hook(forward_hook)

    # 可选：启用 PyTorch 的异常检测（会减慢速度，但能捕获 NaN/Inf）
    # torch.autograd.set_detect_anomaly(True)

    with torch.no_grad():
        # ========== 关键修改：兼容模型返回值个数 ==========
        output = model(x, t, cond)
        if isinstance(output, tuple):
            # 模型返回了多个值
            out = output[0]
            new_state = output[1] if len(output) > 1 else None
        else:
            # 模型只返回一个值
            out = output
            new_state = None

    print("前向传播完成。")
    print(f"输出形状: {out.shape}")
    print(f"状态形状: {new_state.shape if new_state is not None else 'None'}")

except Exception as e:
    print("异常捕获：")
    traceback.print_exc()
    sys.exit(1)