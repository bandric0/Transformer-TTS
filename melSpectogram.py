import torch
import torchaudio
from torchaudio.functional import spectrogram
from params import p


spec_transform = torchaudio.transforms.Spectrogram(n_fft = p.n_fft, win_length = p.win_length, hop_length = p.hop_length, power = p.power)
mel_scale_transform = torchaudio.transforms.MelScale(n_mels = p.mel_freq, sample_rate = p.sr, n_stft = p.n_stft)
mel_inverse_transform = torchaudio.transforms.InverseMelScale(n_mels = p.mel_freq, sample_rate = p.sr, n_stft = p.n_stft).cuda()
griffnlim_transform = torchaudio.transforms.GriffinLim(n_fft = p.n_fft, win_length = p.win_length, hop_length = p.hop_length).cuda()

def norm_mel_spec_db(mel_spec):  
    mel_spec = ((2.0*mel_spec - p.min_level_db) / (p.max_db/p.norm_db)) - 1.0
    mel_spec = torch.clip(mel_spec, -p.ref*p.norm_db, p.ref*p.norm_db)
    return mel_spec


def denorm_mel_spec_db(mel_spec):
    mel_spec = (((1.0 + mel_spec) * (p.max_db/p.norm_db)) + p.min_level_db) / 2.0 
    return mel_spec


def pow_to_db_mel_spec(mel_spec):
    mel_spec = torchaudio.functional.amplitude_to_DB(
        mel_spec,
        multiplier = p.ampl_multiplier, 
        amin = p.ampl_amin, 
        db_multiplier = p.db_multiplier, 
        top_db = p.max_db
    )
    mel_spec = mel_spec/p.scale_db
    return mel_spec


def db_to_power_mel_spec(mel_spec):
    mel_spec = mel_spec*p.scale_db
    mel_spec = torchaudio.functional.DB_to_amplitude(
        mel_spec,
        ref=p.ampl_ref,
        power=p.ampl_power
    )  
    return mel_spec

def convert_to_mel_spec(wav):
    spec = spec_transform(wav)
    mel_spec = mel_scale_transform(spec)
    db_mel_spec = pow_to_db_mel_spec(mel_spec)
    db_mel_spec = db_mel_spec.squeeze(0)
    return db_mel_spec


def inverse_mel_spec_to_wav(mel_spec):
    power_mel_spec = db_to_power_mel_spec(mel_spec)
    spectrogram = mel_inverse_transform(power_mel_spec)
    pseudo_wav = griffnlim_transform(spectrogram)
    return pseudo_wav

if __name__ == "__main__":
    wav_path = "data/LJSpeech-1.1/wavs/LJ001-0001.wav"
    wav, sr = torchaudio.load(wav_path, normalize=True)
    
    mel_spec = convert_to_mel_spec(wav)
    print("Mel Spectrogram Shape:", mel_spec.shape)

    reconstructed_wav = inverse_mel_spec_to_wav(mel_spec.cuda())
    print("Reconstructed Waveform Shape:", reconstructed_wav.shape)

    torchaudio.save("test.wav", reconstructed_wav.cpu(), sample_rate=p.sr)