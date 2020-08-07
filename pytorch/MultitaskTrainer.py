
from utils.vocab import Vocab
from utils.util import epoch_time, initialize_weights, set_seed, count_parameters
import utils.constants as constants
from utils.loss import LabelSmoothing, LossCompute
from utils.optimizer import NoamOpt

from Dataloader import ParallelDataset, get_dataloader

from models.Multitask import Multitask
from layers.Encoder import Encoder
from layers.Decoder import Decoder

from queue import PriorityQueue

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

def load_model(args, source_vocabs, target_vocabs, device, max_length):
	if args.load_encoder:
		from collections import OrderedDict
		encoder = OrderedDict()
		model = torch.load(args.model)
		for item in model:
			if item.startswith("encoder"):
				encoder[item.replace("encoder.","")] = model[item]
		print("Building an model using a pre-trained encoder ... ")
		current = build_model(args, source_vocabs, target_vocabs, device, max_length, encoder)
		return current
	else:
		mtl = build_model(args, source_vocabs, target_vocabs, device, max_length)
		mtl.load_state_dict(torch.load(args.model))
		print("Building an model using the encoder and the decoder ... ")
		return mtl

def build_model(args, source_vocabs, target_vocabs, device, max_length , encoder=None):

	input_dim = source_vocabs[0].len()
	enc = Encoder(input_dim, 
		args.hidden_size, 
		args.encoder_layer, 
		args.encoder_head, 
		args.encoder_ff_size, 
		args.encoder_dropout, 
		device,
     		max_length=max_length).to(device)
	if encoder is None:
		enc.apply(initialize_weights);
	else:
		enc.load_state_dict(encoder)

	decs = []

	for target_vocab in target_vocabs:

		output_dim = target_vocab.len()
		dec = Decoder(output_dim, 
				args.hidden_size, 
				args.decoder_layer, 
				args.decoder_head, 
				args.decoder_ff_size, 
				args.decoder_dropout, 
				device,
        max_length=max_length).to(device)

		if args.tie_embeddings:
			dec.tok_embedding = enc.tok_embedding
			dec.fc_out.weight = enc.tok_embedding.weight

		dec.apply(initialize_weights);
		decs.append(dec)

	model = Multitask(enc, decs, constants.PAD_IDX, constants.PAD_IDX, device).to(device)

	return model

def train_step(model, loader, loss_compute, device, task_id = 0):

	model.train()

	(src, tgt) = next(iter(loader))

	n_tokens = (torch.flatten(src != 1)).sum(dim=0) + (torch.flatten(tgt != 1)).sum(dim=0)

	src = src.to(device)
	tgt = tgt.to(device)

	output, _ = model(src, tgt[:,:-1], task_id=task_id)        
	#output = [batch size, tgt len - 1, output dim]
	#tgt = [batch size, tgt len]
	output_dim = output.shape[-1]
	output = output.contiguous().view(-1, output_dim)
	tgt = tgt[:,1:].contiguous().view(-1)
	#output = [batch size * tgt len - 1, output dim]
	#tgt = [batch size * tgt len - 1]

	loss = loss_compute(output, tgt) # , norm = n_tokens 1000

	return loss #/ n_tokens


def evaluate(model, loader, loss_compute, device, task_id=0):
    
	model.eval()  
	epoch_loss = 0
	total_tokens = 0
	with torch.no_grad():

		for i, (src, tgt) in enumerate(loader):

			n_tokens = (torch.flatten(src != 1)).sum(dim=0) + (torch.flatten(tgt != 1)).sum(dim=0)			

			src = src.to(device)
			tgt = tgt.to(device)
			output, _ = model(src, tgt[:,:-1], task_id=task_id)
			#output = [batch size, tgt len - 1, output dim]
			#tgt = [batch size, tgt len]
			output_dim = output.shape[-1]
			output = output.contiguous().view(-1, output_dim)
			tgt = tgt[:,1:].contiguous().view(-1)

			#output = [batch size * tgt len - 1, output dim]
			#tgt = [batch size * tgt len - 1]

			loss = loss_compute(output, tgt) #, n_tokens
			epoch_loss += loss

		if torch.equal(model.decoders[task_id].fc_out.weight, model.encoder.tok_embedding.weight):
			print("decoder output and encoder embeddings are the same")

	return epoch_loss / len(loader) #total_tokens



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
			output, attention = model.decoders[task_id](trg_tensor, enc_src, trg_mask, src_mask)

		pred_token = output.argmax(2)[:,-1].item()
		trg_indexes.append(pred_token)

		if pred_token == constants.EOS_IDX:
			break

	trg_tokens = [target_vocab.itos(i) for i in trg_indexes]

	return ' '.join(trg_tokens[1:])



class BeamSearchNode(object):
	def __init__(self, previousNode, wordId, logProb, length):

		self.prevNode = previousNode
		self.wordid = wordId
		self.logp = logProb
		self.leng = length

	def eval(self, alpha=1.0):
		reward = 0
		# Add here a function for shaping a reward
		return self.logp / float(self.leng - 1 + 1e-6) #+ alpha * reward


def translate_sentence_beam(model, task_id, sentence, source_vocab, target_vocab, device, beam_size = 5, max_length = 180):

	model.eval()

	topk = 1

	tokens = [token.lower() for token in sentence.split()]
	tokens = [constants.SOS_STR] + tokens + [constants.EOS_STR]

	if len(tokens) < max_length:
		tokens = tokens + [constants.PAD_STR for _ in range(max_length - len(tokens))]
	else:
		tokens = tokens[:max_length-1] + [constants.EOS_STR]

	src_indexes = [source_vocab.stoi(token) for token in tokens]
	src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
	src_mask = model.make_src_mask(src_tensor)

	enc_src = None
	with torch.no_grad():
		enc_src = model.encoder(src_tensor, src_mask)

	trg_indexes = [constants.SOS_IDX]

	# Number of sentence to generate
	endnodes = []
	number_required = min((topk + 1), topk - len(endnodes))

	# starting node -  hidden vector, previous node, word id, logp, length
	node = BeamSearchNode(None, trg_indexes, 0, 1)
	nodes = PriorityQueue()

	# start the queue
	nodes.put((-node.eval(), node))
	qsize = 1

	while True:
		# give up when decoding takes too long
		if qsize > 2000: break

		# fetch the best node
		score, n = nodes.get()
		trg_indexes = n.wordid

		if n.wordid[-1] == constants.EOS_IDX and n.prevNode != None:
			endnodes.append((score, n))
			# if we reached maximum # of sentences required
			if len(endnodes) >= number_required:
				break
			else:
				continue


		trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
		trg_mask = model.make_trg_mask(trg_tensor)

		with torch.no_grad():
			# decode for one step using decoder
			output, attention = model.decoders[task_id](trg_tensor, enc_src, trg_mask, src_mask)

		# PUT HERE REAL BEAM SEARCH OF TOP
		log_prob, indexes = torch.topk(output, beam_size)
		nextnodes = []

		for lp, idx in zip(log_prob, indexes):
			print(idx, "\t", lp)


		for new_k in range(beam_size):
			decoded_t = indexes[0][new_k].view(1, -1)
			log_p = log_prob[0][new_k].item()

			node = BeamSearchNode(trg_indexes, trg_indexes.append(decoded_t), n.logp + log_p, n.leng + 1)
			score = -node.eval()
			nextnodes.append((score, node))

		# put them into queue
		for i in range(len(nextnodes)):
			score, nn = nextnodes[i]
			nodes.put((score, nn))
			# increase qsize
		qsize += len(nextnodes) - 1

	print(endnodes)

		
def train(args):

	set_seed(args.seed)

	device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

	batch_size = args.batch_size
	max_length = args.max_length
	mtl = args.mtl

	learning_rate = 0.0005
	if not args.learning_rate:
		learning_rate = args.learning_rate

	if len(args.train_source) != len(args.train_target):
		print("Error.Number of inputs in train are not the same")
		return

	if len(args.dev_source) != len(args.dev_target):
		print("Error: Number of inputs in dev are not the same")
		return

	if not args.tie_embeddings:
		print("Building Encoder vocabulary")
		source_vocabs = build_vocab(args.train_source, args.src_vocab, save_dir=args.save_dir)
		print("Building Decoder vocabulary")
		target_vocabs = build_vocab(args.train_target, args.tgt_vocab, mtl=mtl, name ="tgt", save_dir=args.save_dir)
	else:
		print("Building Share vocabulary")
		source_vocabs = build_vocab(args.train_source + args.train_target, args.src_vocab, name="tied", save_dir=args.save_dir)
		if mtl:
			target_vocabs = [source_vocabs[0] for _ in range(len(args.train_target))]
		else:
			target_vocabs = source_vocabs
	print("Number of source vocabularies:", len(source_vocabs))
	print("Number of target vocabularies:", len(target_vocabs))

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

	if args.model is not None:
		print("Loading the encoder from an external model...")
		multitask_model = load_model(args, source_vocabs, target_vocabs, device, max_length)
	else:
		print("Building model")
		multitask_model = build_model(args, source_vocabs, target_vocabs, device, max_length)

	print(f'The Transformer has {count_parameters(multitask_model):,} trainable parameters')
	print(f'The Encoder has {count_parameters(multitask_model.encoder):,} trainable parameters')
	for index, decoder in enumerate(multitask_model.decoders):
		print(f'The Decoder {index+1} has {count_parameters(decoder):,} trainable parameters')


	# Defining CrossEntropyLoss as default
	#criterion = nn.CrossEntropyLoss(ignore_index = constants.PAD_IDX)
	criterions = [LabelSmoothing(size=target_vocab.len(), padding_idx=constants.PAD_IDX, smoothing=0.1) \
                                        for target_vocab in target_vocabs]

	# Default optimizer
	optimizer = torch.optim.Adam(multitask_model.parameters(), lr = learning_rate, betas=(0.9, 0.98), eps=1e-09)
	model_opts = [NoamOpt(args.hidden_size, args.warmup_steps, optimizer) for _ in target_vocabs]

	task_id = 0
	print_loss_total = 0  # Reset every print_every

	n_tasks = len(train_loaders)
	best_valid_loss = [float('inf') for _ in range(n_tasks)]
	'''
	patience = 30
	if not args.patience:
		patience = args.patience

	if n_tasks > 1:
		print("Patience wont be taking into account in Multitask learning")

	for _iter in range(1, args.steps + 1):

		train_loss = train_step(multitask_model, train_loaders[task_id], \
                       LossCompute(criterions[task_id], model_opts[task_id]), device, task_id = task_id)

		print_loss_total += train_loss

		if _iter % args.print_every == 0:
			print_loss_avg = print_loss_total / args.print_every
			print_loss_total = 0
			print(f'Task: {task_id:d} | Step: {_iter:d} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')


		if _iter % args.eval_steps == 0:
			print("Evaluating...")
			valid_loss = evaluate(multitask_model, dev_loaders[task_id], LossCompute(criterions[task_id], None), \
                            device, task_id=task_id)
			print(f'Task: {task_id:d} | Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
			if valid_loss < best_valid_loss[task_id]:
				print(f'The loss decreased from {best_valid_loss[task_id]:.3f} to {valid_loss:.3f} in the task {task_id}... saving checkpoint')
				patience = 30
				best_valid_loss[task_id] = valid_loss
				torch.save(multitask_model.state_dict(), args.save_dir + 'model.pt')
				print("Saved model.pt")
			else:
				if n_tasks == 1:
					if patience == 0:
						break
					else:
						patience -= 1

			if n_tasks > 1:
				print("Changing to the next task ...")
				task_id = (0 if task_id == n_tasks - 1 else task_id + 1)

	'''
	multitask_model.load_state_dict(torch.load(args.save_dir + 'model.pt'))


	print("Evaluating and testing")
	for index, eval_name in enumerate(args.eval):
		n = len(eval_name.split("/"))
		name = eval_name.split("/")[n-1]
		print(f'Reading {eval_name}')
		fout = open(args.save_dir + name + "." + str(index) + ".out", "w")
		with open(eval_name, "r") as f:
			for sentence in f:
				#output = translate_sentence(multitask_model, index, sentence, source_vocabs[0], target_vocabs[index], device, max_length)
				output = ""
				translate_sentence_beam(multitask_model, index, sentence, source_vocabs[0], target_vocabs[index], device, max_length=max_length)
				fout.write(output.replace("<eos>","").strip() + "\n")
		fout.close()
	'''
	for index, test_name in enumerate(args.test):
		n = len(test_name.split("/"))
		name = test_name.split("/")[n-1]
		print(f'Reading {test_name}')
		fout = open(args.save_dir + name + "." + str(index) + ".out", "w")
		with open(test_name, "r") as f:
			for sentence in f:
				output = translate_sentence(multitask_model, index, sentence, source_vocabs[0], target_vocabs[index], device, max_length)
				fout.write(output.replace("<eos>","").strip() + "\n")
		fout.close()
	'''			

