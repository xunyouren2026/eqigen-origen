# models/text_encoder.py
import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    def __init__(self, encoder_type, model_name, device):
        super().__init__()
        self.encoder_type = encoder_type
        self.device = device
        if encoder_type == "clip" and model_name == "openai/clip-vit-large-patch14":
            local_path = "./models/clip"
            self.tokenizer = CLIPTokenizer.from_pretrained(
                local_path, local_files_only=True)
            self.text_encoder = CLIPTextModel.from_pretrained(
                local_path, local_files_only=True).to(device)
        elif encoder_type == "clip":
            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_name, local_files_only=True)
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_name, local_files_only=True).to(device)
        elif encoder_type == "t5":
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True)
            self.text_encoder = AutoModel.from_pretrained(
                model_name, local_files_only=True).to(device)
        elif encoder_type == "xlm-roberta":
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True)
            self.text_encoder = AutoModel.from_pretrained(
                model_name, local_files_only=True).to(device)
        else:
            raise ValueError(f"Unknown text encoder: {encoder_type}")

    def forward(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state
