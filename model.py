import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from params import p
from dataset import maskFromSeqLengths

class TTSLoss(torch.nn.Module):
    def __init__(self):
        super(TTSLoss, self).__init__()
        
        self.mse_loss = torch.nn.MSELoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, mel_postnet_out, mel_out, stop_token_out, mel_target, stop_token_target):      
        stop_token_target = stop_token_target.view(-1, 1)

        stop_token_out = stop_token_out.view(-1, 1)
        mel_loss = self.mse_loss(mel_out, mel_target) + self.mse_loss(mel_postnet_out, mel_target)

        stop_token_loss = self.bce_loss(stop_token_out, stop_token_target) * p.r_gate

        return mel_loss + stop_token_loss


class EncoderBlock(nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape=p.embedding_size)
        self.attn = torch.nn.MultiheadAttention(embed_dim=p.embedding_size, num_heads=4, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(p.embedding_size, p.dim_feedforward),
            nn.ReLU(),
            nn.Linear(p.dim_feedforward, p.embedding_size),
        )
      
    def forward(self, x, attn_mask = None, key_padding_mask = None):
        x_out = self.norm(x)
        x_out, _ = self.attn(query=x_out, key=x_out, value=x_out,attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + x_out    

        x = self.ffn(x)
        x = x + x_out
        
        return x


class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape=p.embedding_size)
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=p.embedding_size, num_heads=4, dropout=0.1, batch_first=True)
        self.attn = torch.nn.MultiheadAttention(embed_dim=p.embedding_size, num_heads=4, dropout=0.1, batch_first=True)    
        self.ffn = nn.Sequential(
            nn.Linear(p.embedding_size, p.dim_feedforward),
            nn.ReLU(),
            nn.Linear(p.dim_feedforward, p.embedding_size),
        )


    def forward(self, x, memory, x_attn_mask = None, x_key_padding_mask = None, memory_attn_mask = None, memory_key_padding_mask = None):
        x_out, _ = self.self_attn(query=x, key=x, value=x, attn_mask=x_attn_mask, key_padding_mask=x_key_padding_mask)
        x = self.norm(x + x_out)
        
        x_out, _ = self.attn(query=x, key=memory, value=memory, attn_mask=memory_attn_mask, key_padding_mask=memory_key_padding_mask)
        x = self.ffn(x + x_out)
        return x


class EncoderPreNet(nn.Module):
    def __init__(self):
        super(EncoderPreNet, self).__init__()
        
        self.embedding = nn.Embedding(
              num_embeddings=p.text_num_embeddings,
              embedding_dim=p.encoder_embedding_size
        )

        self.linear = nn.Linear(
            p.encoder_embedding_size, 
            p.embedding_size
        )
         
        self.dropout = torch.nn.Dropout(0.5)
  

    def forward(self, text):
        x = self.embedding(text)
        x = self.linear(x)

        x = x.transpose(2, 1) 

        x = F.relu(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)
        
        return x


class PostNet(nn.Module):
    def __init__(self):
        super(PostNet, self).__init__()  
        
        self.conv1 = nn.Conv1d(
            p.mel_freq, 
            p.postnet_embedding_size,
            kernel_size=p.postnet_kernel_size, 
            stride=1,
            padding=int((p.postnet_kernel_size - 1) / 2), 
            dilation=1
        )
        self.bn1 = nn.BatchNorm1d(p.postnet_embedding_size)
        self.dropout1 = torch.nn.Dropout(0.5)

        self.conv2 = nn.Conv1d(
            p.postnet_embedding_size, 
            p.mel_freq,
            kernel_size=p.postnet_kernel_size, 
            stride=1,
            padding=int((p.postnet_kernel_size - 1) / 2), 
            dilation=1
        )
        self.bn2 = nn.BatchNorm1d(p.mel_freq)
        self.dropout2 = torch.nn.Dropout(0.5)


    def forward(self, x):
        x = x.transpose(2, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.tanh(x)
        x = self.dropout1(x)


        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x = x.transpose(1, 2)

        return x


class DecoderPreNet(nn.Module):
    def __init__(self):
        super(DecoderPreNet, self).__init__()
        self.linear_1 = nn.Linear(
            p.mel_freq, 
            p.embedding_size
        )

        self.linear_2 = nn.Linear(
            p.embedding_size, 
            p.embedding_size
        )

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        
        x = F.dropout(x, p=0.5, training=True)

        x = self.linear_2(x)
        x = F.relu(x)    
        x = F.dropout(x, p=0.5, training=True)

        return x    


class TransformerTTS(nn.Module):
    def __init__(self, device="cuda"):
        super(TransformerTTS, self).__init__()

        self.encoder_prenet = EncoderPreNet()
        self.decoder_prenet = DecoderPreNet()
        self.postnet = PostNet()

        self.pos_encoding = nn.Embedding(
            num_embeddings=p.max_mel_time, 
            embedding_dim=p.embedding_size
        )

        self.encoder_block = EncoderBlock()
        self.decoder_block = DecoderBlock()

        self.linear_1 = nn.Linear(p.embedding_size, p.mel_freq) 
        self.linear_2 = nn.Linear(p.embedding_size, 1)

        self.norm_memory = nn.LayerNorm(
            normalized_shape=p.embedding_size
        )


    def forward(self, text, text_len, mel, mel_len):  
        N = text.shape[0]
        S = text.shape[1]
        TIME = mel.shape[1]

        self.src_key_padding_mask = torch.zeros((N, S), device=text.device).masked_fill(~maskFromSeqLengths(text_len, max_length=S), float("-inf"))
        self.src_mask = torch.zeros((S, S),device=text.device).masked_fill(torch.triu(torch.full((S, S), True, dtype=torch.bool), diagonal=1).to(text.device), float("-inf"))
        self.tgt_key_padding_mask = torch.zeros((N, TIME), device=mel.device).masked_fill(~maskFromSeqLengths(mel_len, max_length=TIME), float("-inf"))
        self.tgt_mask = torch.zeros((TIME, TIME), device=mel.device).masked_fill(torch.triu(torch.full((TIME, TIME), True, device=mel.device, dtype=torch.bool), diagonal=1), float("-inf"))
        self.memory_mask = torch.zeros((TIME, S), device=mel.device).masked_fill(torch.triu(torch.full((TIME, S), True, device=mel.device, dtype=torch.bool), diagonal=1), float("-inf"))    

        text_x = self.encoder_prenet(text)  
        
        pos_codes = self.pos_encoding(torch.arange(p.max_mel_time).to(mel.device))

        S = text_x.shape[1]
        text_x = text_x + pos_codes[:S]
        
        text_x = self.encoder_block(text_x, attn_mask = self.src_mask, key_padding_mask = self.src_key_padding_mask)
        text_x = self.norm_memory(text_x)
            
        mel_x = self.decoder_prenet(mel) 
        mel_x = mel_x + pos_codes[:TIME]
        

        mel_x = self.decoder_block(x=mel_x, memory=text_x, x_attn_mask=self.tgt_mask,  x_key_padding_mask=self.tgt_key_padding_mask, memory_attn_mask=self.memory_mask, memory_key_padding_mask=self.src_key_padding_mask)

        mel_linear = self.linear_1(mel_x)
        mel_postnet = self.postnet(mel_linear)
        mel_postnet = mel_linear + mel_postnet
        stop_token = self.linear_2(mel_x)

        bool_mel_mask = self.tgt_key_padding_mask.ne(0).unsqueeze(-1).repeat(1, 1, p.mel_freq)
        mel_linear = mel_linear.masked_fill(bool_mel_mask, 0)
        mel_postnet = mel_postnet.masked_fill(bool_mel_mask, 0)
        stop_token = stop_token.masked_fill(bool_mel_mask[:, :, 0].unsqueeze(-1), 1e3).squeeze(2)
        
        return mel_postnet, mel_linear, stop_token 

    @torch.no_grad()
    def inference(self, text, max_length=800, stop_token_threshold = 0.5, with_tqdm = True):
        self.eval()    
        self.train(False)
        text_lengths = torch.tensor(text.shape[1]).unsqueeze(0).cuda()
        
        mel_padded = torch.zeros((1, 1, p.mel_freq), device="cuda")
        mel_lengths = torch.tensor(1).unsqueeze(0).cuda()
        stop_token_outputs = torch.FloatTensor([]).to(text.device)

        if with_tqdm:
            iters = tqdm(range(max_length))
        else:
            iters = range(max_length)

        for _ in iters:
            mel_postnet, mel_linear, stop_token = self(text, text_lengths, mel_padded, mel_lengths)
            mel_padded = torch.cat([mel_padded, mel_postnet[:, -1:, :]], dim=1)
          
            if torch.sigmoid(stop_token[:,-1]) > stop_token_threshold:      
                break
            else:
                stop_token_outputs = torch.cat([stop_token_outputs, stop_token[:,-1:]], dim=1)
                mel_lengths = torch.tensor(mel_padded.shape[1]).unsqueeze(0).cuda()

        return mel_postnet, stop_token_outputs
