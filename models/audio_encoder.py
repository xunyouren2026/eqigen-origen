import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", device='cuda'):
        super().__init__()
        # 本地路径
        local_path = "./models/wav2vec2-base"
        # 强制只使用本地文件，避免联网
        self.model = Wav2Vec2Model.from_pretrained(
            local_path, local_files_only=True).to(device)
        self.processor = Wav2Vec2Processor.from_pretrained(
            local_path, local_files_only=True)
        self.device = device

    def load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    def forward(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        input_values = self.processor(waveform.squeeze(
            0), return_tensors="pt", sampling_rate=16000).input_values
        input_values = input_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_values)
        return outputs.last_hidden_state
