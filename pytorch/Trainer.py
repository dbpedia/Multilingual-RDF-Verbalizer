
from utils.vocab import Vocab
from utils.util import epoch_time
import utils.constants as constants

from Dataloader import ParallelDataset, get_dataloader

from models.Sequence2Sequence import Seq2Seq
from layers.Encoder import Encoder
from layers.Decoder import Decoder

import torch
import torch.nn as nn

import numpy as np
import random
import math
import time

seed = 13

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True



def build_vocab(source_files, target_files, mtl=False):
	source_vocabs = []
	target_vocabs = []

	print("Joining all vocabularies in the encoder")
	source_vocab = Vocab()
	source_vocab.build_vocab(source_files)
	source_vocabs.append(source_vocab)

	if mtl is True:
		for index, target_file in enumerate(target_files):
			print("Building the vocabulary", (index + 1) ,"in the decoder")
			target_vocab = Vocab()
			target_vocab.build_vocab([target_file])
			target_vocabs.append(target_vocab)
	else:
		print("Joining all vocabularies in the decoder")
		target_vocab = Vocab()
		target_vocab.build_vocab(target_files)
		target_vocabs.append(target_vocab)

	return source_vocabs, target_vocabs


def build_dataset(source_files, target_files, batch_size, shuffle=False, \
			source_vocabs = None, target_vocabs =None, mtl=False):
	loaders = []

	for index, (source_file, target_file) in enumerate(zip(source_files, target_files)):

		if mtl is True:
			_set = ParallelDataset(source_file, target_file, max_length = max_length, \
									source_vocab = source_vocabs[0], target_vocab = target_vocabs[index])
		else:
			_set = ParallelDataset(source_file, target_file, max_length = max_length, \
									source_vocab = source_vocabs[0], target_vocab = target_vocabs[0])

		loader = get_dataloader(_set, batch_size, shuffle=shuffle)
		loaders.append(loader)
	return loaders

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def build_model(source_vocabs, target_vocabs, device):

	HID_DIM = 256
	ENC_LAYERS = 3
	DEC_LAYERS = 3
	ENC_HEADS = 8
	DEC_HEADS = 8
	ENC_PF_DIM = 512
	DEC_PF_DIM = 512
	ENC_DROPOUT = 0.1
	DEC_DROPOUT = 0.1

	INPUT_DIM = source_vocabs[0].len()
	enc = Encoder(INPUT_DIM, 
			HID_DIM, 
			ENC_LAYERS, 
			ENC_HEADS, 
			ENC_PF_DIM, 
			ENC_DROPOUT, 
			device)
	enc.apply(initialize_weights);

	models = []

	for target_vocab in target_vocabs:

		OUTPUT_DIM = target_vocab.len()

		dec = Decoder(OUTPUT_DIM, 
				HID_DIM, 
				ENC_LAYERS, 
				ENC_HEADS, 
				ENC_PF_DIM, 
				ENC_DROPOUT, 
				device)
		dec.apply(initialize_weights);

		model = Seq2Seq(enc, dec, constants.PAD_IDX, constants.PAD_IDX, device).to(device)
		models.append(model)

	return models


def train(model, loader, optimizer, criterion, clip):

	model.train()

	_loss = 0

	#for i, (src, tgt) in enumerate(loader):
	(src, tgt) = next(iter(loader))
	optimizer.zero_grad()
	output, _ = model(src, tgt[:,:-1])        
	#output = [batch size, tgt len - 1, output dim]
	#tgt = [batch size, tgt len]
	output_dim = output.shape[-1]
	output = output.contiguous().view(-1, output_dim)
	tgt = tgt[:,1:].contiguous().view(-1)
	#output = [batch size * tgt len - 1, output dim]
	#tgt = [batch size * tgt len - 1]

	loss = criterion(output, tgt)
	loss.backward()

	torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

	optimizer.step()
	_loss += loss.item()

	return _loss


def evaluate(model, loader, criterion):
    
	model.eval()  
	epoch_loss = 0
	with torch.no_grad():

		for i, (src, tgt) in enumerate(loader):

			output, _ = model(src, tgt[:,:-1])
			#output = [batch size, tgt len - 1, output dim]
			#tgt = [batch size, tgt len]
			output_dim = output.shape[-1]
			output = output.contiguous().view(-1, output_dim)
			tgt = tgt[:,1:].contiguous().view(-1)

			#output = [batch size * tgt len - 1, output dim]
			#tgt = [batch size * tgt len - 1]

			loss = criterion(output, trg)
			epoch_loss += loss.item()

	return epoch_loss / len(iterator)




#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
batch_size = 128
max_length = 100
mtl = True

LEARNING_RATE = 0.0005


criterion = nn.CrossEntropyLoss(ignore_index = constants.PAD_IDX)


dev_source_files = ["data/ordering/dev.src", "data/structing/dev.src", "data/lexicalization/dev.src"]
dev_target_files = ["data/ordering/dev.trg", "data/structing/dev.trg", "data/lexicalization/dev.trg"]

#dev_source_files = ["data/ordering/dev.src"]
#dev_target_files = ["data/ordering/dev.trg"]

train_source_files = ["data/ordering/train.src", "data/structing/train.src", "data/lexicalization/train.src"]
train_target_files = ["data/ordering/train.trg", "data/structing/train.trg", "data/lexicalization/train.trg"]

#train_source_files = ["data/structing/train.src"]
#train_target_files = ["data/structing/train.trg"]


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')


source_vocabs, target_vocabs = build_vocab(train_source_files, train_target_files, mtl=mtl)

train_loaders = build_dataset(train_source_files, train_target_files, batch_size, \
			source_vocabs=source_vocabs, target_vocabs=target_vocabs, shuffle=True, mtl=mtl)

dev_loaders = build_dataset(dev_source_files, dev_target_files, batch_size, \
			source_vocabs=source_vocabs, target_vocabs=target_vocabs, mtl=mtl)

models = build_model(source_vocabs, target_vocabs, device)

optimizers = []
for model in models:
	optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
	optimizers.append(optimizer)

steps = 10000
print_every = 100
evaluation_step = 500

model_idx = 0

print_loss_total = 0  # Reset every print_every

for _iter in range(1, steps + 1):




	train_loss = train(models[0], train_loaders[0], optimizers[0], criterion, CLIP)
	print_loss_total += train_loss

	if _iter % print_every == 0:
		print_loss_avg = print_loss_total / print_every
		print_loss_total = 0
		print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))


'''
while steps > 0:

	model_idx = 0

	start_time = time.time()

	train_loss = train(models[model_idx], train_loaders[model_idx], optimizers[model_idx], criterion, CLIP)
	valid_loss = evaluate(models[model_idx], dev_loaders[model_idx], criterion)

	end_time = time.time()

	epoch_mins, epoch_secs = epoch_time(start_time, end_time)

	if valid_loss < best_valid_loss:
		best_valid_loss = valid_loss
		torch.save(model.state_dict(), 'tut6-model.pt')

	print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
	print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
	print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

'''



		




