import torch
import torch.nn as nn
import torch.nn.functional as F
from params import p

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(p.embedding_size, 4, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(p.embedding_size),
            nn.Linear(p.embedding_size, p.dim_feedforward),
            nn.ReLU(),
            nn.Linear(p.dim_feedforward, p.embedding_size),
            nn.Dropout(0.1)
        )

    def forward(self, x, mask):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        return self.ffn(attn_out + x)

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(p.embedding_size, 4, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(p.embedding_size),
            nn.Linear(p.embedding_size, p.dim_feedforward),
            nn.ReLU(),
            nn.Linear(p.dim_feedforward, p.embedding_size),
            nn.Dropout(0.1)
        )

    def forward(self, x, memory, mask):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        cross_attn_out, _ = self.attn(attn_out, memory, memory)
        return self.ffn(cross_attn_out + x)

class TransformerTTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(p.text_num_embeddings, p.encoder_embedding_size),
            nn.Conv1d(p.encoder_embedding_size, p.encoder_embedding_size, 3, padding=1),
            EncoderBlock(), EncoderBlock()
        )
        self.decoder = nn.Sequential(
            nn.Linear(p.mel_freq, p.embedding_size),
            DecoderBlock(), DecoderBlock(),
            nn.Linear(p.embedding_size, p.mel_freq)
        )
        self.postnet = nn.Conv1d(p.mel_freq, p.mel_freq, 5, padding=2)

    def forward(self, text, mel):
        encoded = self.encoder(text)
        decoded = self.decoder(mel, encoded, mask=None)
        return self.postnet(decoded)

