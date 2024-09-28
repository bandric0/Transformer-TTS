import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd

from params import p

from textTransform import textToSeq
from melSpectogram import convert_to_mel_spec

class TTSDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.memo = {}

    def __getitem__(self, index):
      row = self.df.iloc[index]
      wav_name = row["wav"]

      text_mel = self.memo.get(wav_name)

      if text_mel is None:
        wav_path = f"data/LJSpeech-1.1/wavs/{wav_name}.wav"
        text = textToSeq(row["text_norm"])

        wav, sr = torchaudio.load(wav_path, normalize=True)
        assert sr == p.sr
        mel = convert_to_mel_spec(wav)
          
        self.memo[wav_name] = (text, mel)

        text_mel = (text, mel)
      
      return text_mel

    def __len__(self):
        return len(self.df)


def maskFromSeqLengths(sequence_lengths, max_length):
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor 



def textMelCollateFn(batch):
  text_length_max = torch.tensor(
    [text.shape[-1] for text, _ in batch], 
    dtype=torch.int32
  ).max()

  mel_length_max = torch.tensor(
    [mel.shape[-1] for _, mel in batch],
    dtype=torch.int32
  ).max()

  
  text_lengths = []
  mel_lengths = []
  texts_padded = []
  mels_padded = []

  for text, mel in batch:
    text_length = text.shape[-1]      

    text_padded = torch.nn.functional.pad(
      text,
      pad=[0, text_length_max-text_length],
      value=0
    )

    mel_length = mel.shape[-1]
    mel_padded = torch.nn.functional.pad(
        mel,
        pad=[0, mel_length_max-mel_length],
        value=0
    )

    text_lengths.append(text_length)    
    mel_lengths.append(mel_length)    
    texts_padded.append(text_padded)    
    mels_padded.append(mel_padded)

  text_lengths = torch.tensor(text_lengths, dtype=torch.int32)
  mel_lengths = torch.tensor(mel_lengths, dtype=torch.int32)
  texts_padded = torch.stack(texts_padded, 0)
  mels_padded = torch.stack(mels_padded, 0).transpose(1, 2)

  stop_token_padded = maskFromSeqLengths(
      mel_lengths,
      mel_length_max
  )
  stop_token_padded = (~stop_token_padded).float()
  stop_token_padded[:, -1] = 1.0
  
  return texts_padded, \
         text_lengths, \
         mels_padded, \
         mel_lengths, \
         stop_token_padded \
    