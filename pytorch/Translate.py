import utils.constants as constants
import torch

from queue import PriorityQueue

def translate(model, task_id, sentences, source_vocab, target_vocab, device, max_length=180, beam_size=None):
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
	def __init__(self, previousNode, wordId, logProb, length):

		self.prevNode = previousNode
		self.wordid = wordId
		self.logp = logProb
		self.leng = length

	def eval(self, alpha=1.0):
		reward = 0
		# Add here a function for shaping a reward
		return self.logp / float(self.leng - 1 + 1e-6) #+ alpha * reward


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


def translate_sentence_beam(model, task_id, sentence, source_vocab, target_vocab, device, beam_size = 5, max_length = 180):

	model.eval()

	tokens = [token.lower() for token in sentence.split()]
	tokens = [constants.SOS_STR] + tokens + [constants.EOS_STR]
	print(' '.join(tokens))
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

		if n.wordid[-1] == constants.EOS_IDX and n.prevNode != None:
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
