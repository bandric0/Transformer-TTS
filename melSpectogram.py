import torch
import torchaudio
from torchaudio.functional import spectrogram
from params import p

spec_transform = torchaudio.transforms.Spectrogram(n_fft = p.n_fft, win_length = p.win_length, hop_length = p.hop_length, power = p.power)
mel_scale_transform = torchaudio.transforms.MelScale(n_mels = p.mel_freq, sample_rate = p.sr, n_stft = p.n_stft)
mel_inverse_transform = torchaudio.transforms.InverseMelScale(n_mels = p.mel_freq, sample_rate = p.sr, n_stft = p.n_stft).cuda()
griffnlim_transform = torchaudio.transforms.GriffinLim(n_fft = p.n_fft, win_length = p.win_length, hop_length = p.hop_length).cuda()

def convert_to_mel_spec(wav):
  spec = spec_transform(wav)
  mel_spec = mel_scale_transform(spec)
  mel_spec = mel_spec.squeeze(0)
  return mel_spec


def inverse_mel_spec_to_wav(mel_spec):
  spectrogram = mel_inverse_transform(mel_spec)
  pseudo_wav = griffnlim_transform(spectrogram)
  return pseudo_wav

if __name__ == "__main__":
    wav_path = "../Downloads/file_example_WAV_2MG.wav"
    wav, sr = torchaudio.load(wav_path, normalize=True)
    
    mel_spec = convert_to_mel_spec(wav)
    print("Mel Spectrogram Shape:", mel_spec.shape)

    reconstructed_wav = inverse_mel_spec_to_wav(mel_spec.cuda())
    print("Reconstructed Waveform Shape:", reconstructed_wav.shape)

    torchaudio.save("test.wav", reconstructed_wav.cpu(), sample_rate=p.sr)