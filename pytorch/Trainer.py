
from utils.vocab import Vocab
from utils.util import epoch_time, initialize_weights, set_seed, count_parameters
import utils.constants as constants

from Dataloader import ParallelDataset, get_dataloader

from models.Sequence2Sequence import Seq2Seq
from layers.Encoder import Encoder
from layers.Decoder import Decoder

import torch
import torch.nn as nn

import math
import time

def build_vocab(files, vocabulary=None, mtl=False, name="src", save_dir="/"):
	vocabs = []

	if vocabulary is not None:
		for v in vocabulary:
			print(f'Loading from {v}')
			vocab = Vocab()
			vocab.load_from_file(v)
			vocabs.append(vocab)
	else:
		if mtl is True:
			for index, f in enumerate(files):
				vocab = Vocab()
				vocab.build_vocab([f])
				vocab.save(save_dir + name + ".vocab." + str(index) + ".json")
				vocabs.append(vocab)
		else:
			vocab = Vocab()
			vocab.build_vocab(files)
			vocab.save(save_dir + name + ".vocab.json")
			vocabs.append(vocab)

	for index, vocab in enumerate(vocabs):
		print(f'vocabulary size {index+1:d}: {vocab.len():d}')

	return vocabs

def build_dataset(source_files, target_files, batch_size, shuffle=False, \
			source_vocabs=None, target_vocabs=None, mtl=False, max_length=180):
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


def build_model(args, source_vocabs, target_vocabs, device, max_length , enc=None):

	'''
	HID_DIM = 256
	ENC_LAYERS = 3
	DEC_LAYERS = 3
	ENC_HEADS = 8
	DEC_HEADS = 8
	ENC_PF_DIM = 512
	DEC_PF_DIM = 512
	ENC_DROPOUT = 0.1
	DEC_DROPOUT = 0.1
	'''

	input_dim = source_vocabs.len()
	enc = Encoder(input_dim, 
		args.hidden_size, 
		args.encoder_layer, 
		args.encoder_head, 
		args.encoder_ff_size, 
		args.encoder_dropout, 
		device,
   		max_length=max_length).to(device)
	enc.apply(initialize_weights);

	output_dim = target_vocabs.len()
	dec = Decoder(output_dim, 
			args.hidden_size, 
			args.decoder_layer, 
			args.decoder_head, 
			args.decoder_ff_size, 
			args.decoder_dropout, 
			device,
       max_length=max_length).to(device)
	dec.apply(initialize_weights);

	model = Seq2Seq(enc, dec, constants.PAD_IDX, constants.PAD_IDX, device).to(device)

	return model


def train_step(model, loader, optimizer, criterion, clip, device, task_id = 0):

	model.train()

	(src, tgt) = next(iter(loader))
	src = src.to(device)
	tgt = tgt.to(device)
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

	return loss.item()


def evaluate(model, loader, criterion, device, task_id=0):
    
	model.eval()  
	epoch_loss = 0
	with torch.no_grad():

		for i, (src, tgt) in enumerate(loader):

			src = src.to(device)
			tgt = tgt.to(device)
			output, _ = model(src, tgt[:,:-1])
			#output = [batch size, tgt len - 1, output dim]
			#tgt = [batch size, tgt len]
			output_dim = output.shape[-1]
			output = output.contiguous().view(-1, output_dim)
			tgt = tgt[:,1:].contiguous().view(-1)

			#output = [batch size * tgt len - 1, output dim]
			#tgt = [batch size * tgt len - 1]

			loss = criterion(output, tgt)
			epoch_loss += loss.item()

	return epoch_loss / len(loader)



def translate_sentence(model, task_id, sentence, source_vocab, target_vocab, device, max_length = 180):

	model.eval()

	tokens = [token.lower() for token in sentence.split()]
	tokens = [constants.SOS_STR] + tokens + [constants.EOS_STR]

	if len(tokens) < max_length:
		tokens = tokens + [constants.PAD_STR for _ in range(max_length - len(tokens))]
	else:
		tokens = tokens[:max_length-1] + [constants.EOS_STR]

	src_indexes = [source_vocab.stoi(token) for token in tokens]
	src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
	src_mask = model.make_src_mask(src_tensor)
    
	with torch.no_grad():
		enc_src = model.encoder(src_tensor, src_mask)

	trg_indexes = [constants.SOS_IDX]

	for i in range(max_length):

		trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
		trg_mask = model.make_trg_mask(trg_tensor)

		with torch.no_grad():
			output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

		pred_token = output.argmax(2)[:,-1].item()
		trg_indexes.append(pred_token)

		if pred_token == constants.EOS_IDX:
			break

	trg_tokens = [target_vocab.itos(i) for i in trg_indexes]

	return ' '.join(trg_tokens[1:])

		
def train(args):

	set_seed(args.seed)

	device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

	batch_size = args.batch_size
	max_length = args.max_length
	mtl = args.mtl
	learning_rate = args.learning_rate

	# Defining CrossEntropyLoss as default
	criterion = nn.CrossEntropyLoss(ignore_index = constants.PAD_IDX)
	clipping = args.gradient_clipping

	#train_source_files = ["data/ordering/train.src", "data/structing/train.src", "data/lexicalization/train.src"]
	#train_target_files = ["data/ordering/train.trg", "data/structing/train.trg", "data/lexicalization/train.trg"]
	#dev_source_files = ["data/ordering/dev.src", "data/structing/dev.src", "data/lexicalization/dev.src"]
	#dev_target_files = ["data/ordering/dev.trg", "data/structing/dev.trg", "data/lexicalization/dev.trg"]

	if len(args.train_source) != len(args.train_target):
		print("Error.Number of inputs in train are not the same")
		return

	if len(args.dev_source) != len(args.dev_target):
		print("Error: Number of inputs in dev are not the same")
		return

	print("Building Encoder vocabulary")
	source_vocabs = build_vocab(args.train_source, args.src_vocab, save_dir=args.save_dir)
	print("Building Decoder vocabulary")
	target_vocabs = build_vocab(args.train_target, args.tgt_vocab, mtl=mtl, name ="tgt", save_dir=args.save_dir)

	# source_vocabs, target_vocabs = build_vocab(args.train_source, args.train_target, mtl=mtl)

	print("Building training set and dataloaders")
	train_loaders = build_dataset(args.train_source, args.train_target, batch_size, \
			source_vocabs=source_vocabs, target_vocabs=target_vocabs, shuffle=True, mtl=mtl, max_length=max_length)
	for train_loader in train_loaders:
		print(f'Train - {len(train_loader):d} batches with size: {batch_size:d}')

	print("Building dev set and dataloaders")
	dev_loaders = build_dataset(args.dev_source, args.dev_target, batch_size, \
			source_vocabs=source_vocabs, target_vocabs=target_vocabs, mtl=mtl, max_length=max_length)
	for dev_loader in dev_loaders:
		print(f'Dev - {len(dev_loader):d} batches with size: {batch_size:d}')

	print("Building model")
	model = build_model(args, source_vocabs[0], target_vocabs[0], device, max_length)
	print(f'The model has {count_parameters(model):,} trainable parameters')

	# Default optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

	task_id = 0
	print_loss_total = 0  # Reset every print_every

	n_tasks = len(train_loaders)
	best_valid_loss = [float('inf') for _ in range(n_tasks)]

	for _iter in range(1, args.steps + 1):

		train_loss = train_step(model, train_loaders[task_id], optimizer, criterion, clipping, device, task_id = task_id)
		print_loss_total += train_loss

		if _iter % args.print_every == 0:
			print_loss_avg = print_loss_total / args.print_every
			print_loss_total = 0  
			print(f'Task: {task_id:d} | Step: {_iter:d} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')


		if _iter % args.eval_steps == 0:
			print("Evaluating...")
			valid_loss = evaluate(model, dev_loaders[task_id], criterion, device, task_id=task_id)
			print(f'Task: {task_id:d} | Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
			if valid_loss < best_valid_loss[task_id]:
				print(f'The loss decreased from {best_valid_loss[task_id]:.3f} to {valid_lost:.3f} in the task {task_id}... saving checkpoint')
				best_valid_loss[task_id] = valid_loss
				torch.save(model.state_dict(), args.save_dir + 'model.pt')
				print("Saved model.pt")

			if n_tasks > 1:
				print("Changing to the next task ...")
				task_id = (0 if task_id == n_tasks - 1 else task_id + 1)


	model.load_state_dict(torch.load(args.save_dir + 'model.pt'))


	print("Evaluating and testing")
	for index, eval_name in enumerate(args.eval):
		n = len(eval_name.split("/"))
		name = eval_name.split("/")[n-1]
		print(f'Reading {eval_name}')
		fout = open(args.save_dir + name + "." + str(index) + ".out", "w")
		with open(eval_name, "r") as f:
			for sentence in f:
				output = translate_sentence(model, index, sentence, source_vocabs[0], target_vocabs[index], device, max_length)
				fout.write(output.replace("<eos>","").strip() + "\n")
		fout.close()

	for index, test_name in enumerate(args.test):
		n = len(test_name.split("/"))
		name = test_name.split("/")[n-1]
		print(f'Reading {test_name}')
		fout = open(args.save_dir + name + "." + str(index) + ".out", "w")
		with open(test_name, "r") as f:
			for sentence in f:
				output = translate_sentence(model, index, sentence, source_vocabs[0], target_vocabs[index], device, max_length)
				fout.write(output.replace("<eos>","").strip() + "\n")
		fout.close()
				

