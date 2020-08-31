import utils.constants as constants
import torch
import numpy as np
import random
from arguments import get_args
import sys

from utils.vocab import Vocab
from utils.util import set_seed, load_params, initialize_weights, build_vocab

from models.Multitask import Multitask
from layers.Encoder import Encoder
from layers.Decoder import Decoder

from queue import PriorityQueue

def translate(model, task_id, sentences, source_vocab, target_vocab, device, max_length=180, beam_size=None):
	'''
		Function that translates a set of triple sets
		model:
		task_id:
		sentences:
		source_vocab:
		target_vocab:
		device:
		max_length:
		beam_size:
	'''

	outputs = []
	if beam_size == 1:
		print("Using greedy search")
		for sentence in sentences:
			outputs.append(translate_sentence(model, task_id, sentence, source_vocab, target_vocab, device,
					 max_length = max_length))
	else:
		print("Using beam search")
		for sentence in sentences:
			outputs.append(translate_sentence_beam(model, task_id, sentence, source_vocab, target_vocab, device,
								 beam_size = beam_size, max_length=max_length))
	return outputs


class BeamSearchNode(object):
	'''
		This class handles the tokens that are being generating in the beam search process
	'''

	def __init__(self, previousNode, wordId, logProb, length):

		self.prevNode = previousNode
		self.wordid = wordId
		self.logp = logProb
		self.leng = length

	def eval(self, alpha=1.0):
		reward = 0
		# Add here a function for shaping a reward
		return self.logp / float(self.leng - 1 + 1e-6) #+ alpha * reward

	def __lt__(self, o):
		return 0


def translate_sentence(model, task_id, sentence, source_vocab, target_vocab, device, max_length = 180):
	'''
		This function translates an specific sentence
		model:
		task_id:
		sentences:
		source_vocab:
		target_vocab:
		device:
		max_length:
	'''

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


def translate_sentence_beam(model, task_id, sentence, source_vocab, target_vocab, device, beam_size = 5, max_length = 180):
	'''
		This function translates an specific sentence
		model:
		task_id:
		sentences:
		source_vocab:
		target_vocab:
		device:
		max_length:
		beam_size:
	'''

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

	enc_src = None
	with torch.no_grad():
		enc_src = model.encoder(src_tensor, src_mask)

	trg_indexes = [constants.SOS_IDX]

	endnodes = []

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

		if (n.wordid[-1] == constants.EOS_IDX and n.prevNode != None) or len(trg_indexes) == max_length:
			endnodes.append((score, n))
			break

		trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
		trg_mask = model.make_trg_mask(trg_tensor)

		with torch.no_grad():
			# decode for one step using decoder
			output, attention = model.decoders[task_id](trg_tensor, enc_src, trg_mask, src_mask)

		log_prob, indexes = torch.topk(output, beam_size)
		nextnodes = []
		for new_k in range(beam_size):
			decoded_t = indexes[0][-1][new_k].item()
			log_p = log_prob[0][-1][new_k].item()

			node = BeamSearchNode(trg_indexes, trg_indexes + [decoded_t], n.logp + log_p, n.leng + 1)
			score = -node.eval()
			nextnodes.append((score, node))

		# put them into queue
		for i in range(len(nextnodes)):
			score, nn = nextnodes[i]
			nodes.put((score, nn))
			# increase qsize
		qsize += len(nextnodes) - 1

	if len(endnodes) == 0:
		score, n = nodes.get()
		trg_indexes = n.wordid
	else:
		trg_indexes = endnodes[0][1].wordid

	trg_tokens = [target_vocab.itos(i) for i in trg_indexes]

	return ' '.join(trg_tokens[1:])


def load_model(model, params, source_vocabs, target_vocabs, device):
	'''
		This method loads a pre-trained model
		args: arguments for loading the model
		params:
		source_vocabs: the source vocabulary for each file
		target_vocabs: the target vocabulary for each file
		device: if use gpu or cpu
		max_length: max length of a sentence
	'''
	print("Loading the model from ", model)
	mtl = build_model(params, source_vocabs, target_vocabs, device)
	mtl.load_state_dict(torch.load(model))
	print("Model loaded")
	return mtl


def build_model(params, source_vocabs, target_vocabs, device):
	'''
		This method builds a model from scratch or using the encoder of a pre-trained model
		args: arguments for loading the model
		source_vocabs: the source vocabulary for each file
		target_vocabs: the target vocabulary for each file
		device: if use gpu or cpu
		max_length: max length of a sentence
		encoder: if the encoder is passed as a pre-trained model
	'''

	input_dim = source_vocabs[0].len()
	enc = Encoder(input_dim, 
		params['hidden_size'], 
		params['encoder_layer'], 
		params['encoder_head'], 
		params['encoder_ff_size'], 
		params['encoder_dropout'], 
		device,
     		max_length=params['max_length']).to(device)
	enc.apply(initialize_weights);

	decs = []

	for target_vocab in target_vocabs:

		output_dim = target_vocab.len()
		dec = Decoder(output_dim, 
				params['hidden_size'], 
				params['decoder_layer'], 
				params['decoder_head'], 
				params['decoder_ff_size'], 
				params['decoder_dropout'], 
				device,
        max_length=params['max_length']).to(device)

		if params['tie_embeddings']:
			dec.tok_embedding = enc.tok_embedding
			dec.fc_out.weight = enc.tok_embedding.weight

		dec.apply(initialize_weights);
		decs.append(dec)

	model = Multitask(enc, decs, constants.PAD_IDX, constants.PAD_IDX, device).to(device)

	return model

def run_translate(model, source_vocab, target_vocabs, save_dir, device, beam_size, filename, max_length, task_id=0):
	'''
		This method builds a model from scratch or using the encoder of a pre-trained model
		model: the model being evaluated
		source_vocabs: the source vocabulary for each file
		target_vocabs: the target vocabulary for each file
		save_dir: path where the outpus will be saved
		beam_size: beam size during the translating
		filenames: filenames of triples to process
		max_length: max length of a sentence
		task_id:
	'''

	print(f'Reading {filename}')
	fout = open(save_dir, "w")
	with open(filename, "r") as f:
		outputs = translate(model, task_id, f, source_vocab, target_vocabs[task_id], device, 
							beam_size=beam_size, max_length=max_length)
		for output in outputs:
			fout.write(output.replace("<eos>","").strip() + "\n")
	fout.close()


def run(args):

	set_seed(args.seed)
	device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

	params = load_params(args.params)


	if not args.src_vocab:
		sys.exit("Source vocab is not found")		

	if not params['tie_embeddings']:
		if not args.src_vocab:
			sys.exit("Target vocab is not found")

		print("Loading Encoder vocabulary")
		source_vocabs = build_vocab(None, args.src_vocab)
		print("Loading Decoder vocabulary")
		target_vocabs = build_vocab(None, args.tgt_vocab)
	else:
		print("Loading Shared vocabulary")
		source_vocabs = build_vocab(None, args.src_vocab)
		if args.mtl:
			target_vocabs = [source_vocabs[0] for _ in range(params["number_decoder"])]
		else:
			target_vocabs = source_vocabs

	print("Number of source vocabularies:", len(source_vocabs))
	print("Number of target vocabularies:", len(target_vocabs))

	if args.model is None:
		sys.exit("Model not found")

	model = load_model(args.model, params, source_vocabs, target_vocabs, device)
	run_translate(model, source_vocabs[0], target_vocabs, args.save_dir, device, 
			args.beam_size, args.input, max_length=params['max_length'], task_id=args.task_id)



if __name__ == "__main__":
	args = get_args()
	global step

	run(args)

