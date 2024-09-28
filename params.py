class Params:
    symbols = [
    'EOS', ' ', '!', ',', '-', '.', \
    ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', \
    'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
    'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'à', \
    'â', 'è', 'é', 'ê', 'ü', '’', '“', '”' \
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']  
    len = 52


    sr = 22050
    n_fft = 2048
    n_stft = 1025
    
    hop_length = 256
    win_length = 1024
    
    mel_freq = 128
    max_mel_time = 1024
    
    max_db = 100  
    scale_db = 10
    ref = 4.0
    power = 2.0
    norm_db = 10 
    ampl_multiplier = 10.0
    ampl_amin = 1e-10
    db_multiplier = 1.0
    ampl_ref = 1.0
    ampl_power = 1.0

    
    
    text_num_embeddings = 2*52
    embedding_size = 64
    encoder_embedding_size = 128 

    dim_feedforward = 256
    postnet_embedding_size = 256

    encoder_kernel_size = 2
    postnet_kernel_size = 4

    
    
    batch_size = 16
    grad_clip = 1.0
    lr = 2.0 * 1e-4
    r_gate = 1.0

    
p = Params()