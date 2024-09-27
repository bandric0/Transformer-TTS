class Params:
    csv_path = "/data/metadata.csv"
    wav_path = "/data/LJSpeech-1.1/wavs"
    
    symbols = [
    'EOS', ' ', '!', ',', '-', '.', \
    ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', \
    'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
    'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'à', \
    'â', 'è', 'é', 'ê', 'ü', '’', '“', '”' \
    ] 
    len = 43

    sr = 22050*2
    n_fft = 1024
    n_stft = 513
    
    hop_length = 64
    mel_freq = 64
    power = 2.0
    win_length = 256

    #sr = 22050
    #n_fft = 2048
    #hop_length = 256
    #mel_freq = 128
    #win_length = 512

p = Params()