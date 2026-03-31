# export.py
import torch
import numpy as np


def export_to_onnx(model, dummy_input, output_path, dynamic_axes=True):
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=13,
        input_names=['input', 'timestep', 'context'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'seq_len', 3: 'height', 4: 'width'},
            'timestep': {0: 'batch_size'},
            'context': {0: 'batch_size', 1: 'context_len'},
            'output': {0: 'batch_size', 2: 'seq_len', 3: 'height', 4: 'width'}
        } if dynamic_axes else None
    )
    print(f"Exported to {output_path}")


def export_to_tensorrt(onnx_path, output_path, fp16=False):
    try:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX file")
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        engine = builder.build_engine(network, config)
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        print(f"Exported to {output_path}")
    except ImportError:
        print("TensorRT not installed. Skipping export.")
        raise


def export_model(model, config, device, output_dir="./exported"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    batch_size = 1
    latent_channels = config.model.vae_latent_channels
    num_frames = config.data.num_frames
    h = config.data.image_size // 8
    w = config.data.image_size // 8
    dummy_input = torch.randn(
        batch_size, latent_channels, num_frames, h, w).to(device)
    dummy_t = torch.randint(
        0, config.model.num_timesteps, (batch_size,)).to(device)
    dummy_context = torch.randn(
        batch_size, 77, config.model.dit_context_dim).to(device)

    onnx_path = os.path.join(output_dir, "model.onnx")
    export_to_onnx(model, (dummy_input, dummy_t, dummy_context), onnx_path)

    trt_path = os.path.join(output_dir, "model.trt")
    try:
        export_to_tensorrt(onnx_path, trt_path,
                           fp16=config.train.mixed_precision)
    except:
        print("TensorRT export failed, skipping.")
