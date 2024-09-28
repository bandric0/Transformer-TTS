import os 
import time

import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from params import p
from dataset import TTSDataset, textMelCollateFn
from model import TTSLoss
from model import TransformerTTS
from melSpectogram import inverse_mel_spec_to_wav
from textTransform import textToSeq


def batch_process(batch):
	text_padded, text_lengths, mel_padded, mel_lengths, stop_token_padded = batch

	text_padded = text_padded.cuda()
	text_lengths = text_lengths.cuda()
	mel_padded = mel_padded.cuda()
	stop_token_padded = stop_token_padded.cuda()
	mel_lengths = mel_lengths.cuda()

	mel_input = torch.cat([torch.zeros((mel_padded.shape[0], 1, p.mel_freq), device=mel_padded.device), mel_padded[:, :-1, :]], dim=1)  

	return text_padded, text_lengths, mel_padded, mel_lengths, mel_input, stop_token_padded



def inference_utterance(model, text):
	sequences = textToSeq(text).unsqueeze(0).cuda()
	postnet_mel, stop_token = model.inference(
		sequences, 
		stop_token_threshold=1e5, 
		with_tqdm = False
	)          
	audio = inverse_mel_spec_to_wav(postnet_mel.detach()[0].T)
				
	fig, (ax1) = plt.subplots(1, 1)
	ax1.imshow(
		postnet_mel[0, :, :].detach().cpu().numpy().T, 
	)
	
	return audio, fig 


def calculate_test_loss(model, test_loader):
	test_loss_mean = 0.0
	model.eval()

	with torch.no_grad():
		for test_i, test_batch in enumerate(test_loader):
			test_text_padded, test_text_lengths, test_mel_padded, test_mel_lengths, test_mel_input, test_stop_token_padded = batch_process(batch)

			test_post_mel_out, test_mel_out, test_stop_token_out = model(test_text_padded, test_text_lengths, test_mel_input, test_mel_lengths)        
			test_loss = criterion(mel_postnet_out = test_post_mel_out, mel_out = test_mel_out, stop_token_out = test_stop_token_out, mel_target = test_mel_padded, stop_token_target = test_stop_token_padded)

			test_loss_mean += test_loss.item()

	test_loss_mean = test_loss_mean / (test_i + 1)  
	return test_loss_mean



def train(training_time):
	df = pd.read_csv("data/LJSpeech-1.1/metadata.csv")
	
	num_rows_to_drop = int(len(df) * 0.9)
	df = df.drop(df.sample(n=num_rows_to_drop).index)

	train_df, test_df = train_test_split(df, test_size=65)
	train_loader = torch.utils.data.DataLoader(TTSDataset(train_df), num_workers=2, shuffle=True, batch_size=p.batch_size, pin_memory=True, drop_last=True, collate_fn=textMelCollateFn)
	test_loader = torch.utils.data.DataLoader(TTSDataset(test_df), num_workers=2, shuffle=True, batch_size=8, pin_memory=True, drop_last=True,  collate_fn=textMelCollateFn)  
    
	train_saved_path = "data/models/best_train_model"
	test_saved_path = "data/models/best_test_model"
  
	criterion = TTSLoss().cuda()
	model = TransformerTTS().cuda()
	optimizer = torch.optim.AdamW(model.parameters(), lr=p.lr)
	scaler = torch.cuda.amp.GradScaler()  

	best_test_loss_mean = float("inf")
	best_train_loss_mean = float("inf")
	
	train_loss_mean = 0.0
	epoch = 0
	i = 0

	first_start_time_sec = time.time()
	start_time_sec = time.time()

	while time.time() - first_start_time_sec < training_time:
		for batch in train_loader:
			text_padded, text_lengths, mel_padded, mel_lengths, mel_input, stop_token_padded = batch_process(batch)

			model.train(True)
			model.zero_grad()

			with torch.autocast(device_type='cuda', dtype=torch.float16):
				post_mel_out, mel_out, stop_token_out = model(text_padded, text_lengths, mel_input, mel_lengths)        
				loss = criterion(mel_postnet_out = post_mel_out, mel_out = mel_out, stop_token_out = stop_token_out, mel_target = mel_padded, stop_token_target = stop_token_padded)

			scaler.scale(loss).backward()      
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), p.grad_clip)
			scaler.step(optimizer)
			scaler.update()	
			
			train_loss_mean += loss.item()


			if i % 100 == 0:
				train_loss_mean = train_loss_mean / p.step_print        
        
				if i % 100 == 0:            
					test_loss_mean = calculate_test_loss(model, test_loader)
					audio, fig = inference_utterance(model, "Hello, World.")

					print(f"{epoch}-{i}) Test loss: {np.round(test_loss_mean, 5)}")

					if i % 100 == 0:
						is_best_train = train_loss_mean < best_train_loss_mean
						is_best_test = test_loss_mean < best_test_loss_mean

						state = {
							"model": model.state_dict(), 
							"optimizer": optimizer.state_dict(), 
							"i": i, 
							"test_loss": test_loss_mean, 
							"train_loss": train_loss_mean
						}

						if is_best_train:
							print(f"{epoch}-{i}) Save best train")
							torch.save(state, train_saved_path)
							best_train_loss_mean = train_loss_mean

						if is_best_test:
							print(f"{epoch}-{i}) Save best test")
							torch.save(state, test_saved_path)
							best_test_loss_mean = test_loss_mean
              

				end_time_sec = time.time()
				time_sec = np.round(end_time_sec - start_time_sec, 3)
				start_time_sec = end_time_sec
				
				print(f"{epoch}-{i}) Train loss: {np.round(train_loss_mean, 5)}; Duration: {time_sec} sec.")
				train_loss_mean = 0.0

			i += 1
		epoch += 1

if __name__ == "__main__":
	train(5*60*60)