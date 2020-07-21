from collections import Counter
import json
import sys

import argparse


parser = argparse.ArgumentParser(description="Getting the embeddings for an specific webnlg dataset")
parser.add_argument(
	'-vocab-prefix','--vocab-prefix', type=str, required=False, help='vocabulary prefix')
parser.add_argument(
	'-data', '--data', type=str, nargs='*', required=True, help='Path to files')
parser.add_argument(
	'-split','--split', action='store_true', required=False, help='Generates a unique vocabulary for all tasks or not')
parser.add_argument(
	'-save-dir','--save_dir', type=str, default="", help='Output directory')


class Vocab(object):

	def __init__(self, lower=True):
		self.vocab = {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}
		self.lower = lower
		self.inverse_vocab = {}

	def build_vocab(self, filenames, min_frequency = 0):
		aux = []

		for filename in filenames:
			with open(filename, "r") as f:
				v = [token for line in f for token in line.split()]
				if self.lower:
					v = [e.lower() for e in v]
				aux = aux + v

		freq = Counter(aux)
		aux = [key for key in freq if freq[key] >= min_frequency]

		for key in aux:
			self.vocab[key] = len(self.vocab)

		self.inverse_vocab = {v: k for k, v in self.vocab.items()}

	def load_from_file(self, name):
		with open(name, "r") as f:
			self.vocab = json.load(f)
			self.inverse_vocab = {v: k for k, v in self.vocab.items()}

	def save(self, name):
		print(f'Saving {name}...')
		with open(name, 'w') as out:
			json.dump(self.vocab, out)	

	def stoi(self, key):

		key = key.lower() if self.lower else key

		if key in self.vocab:
			return self.vocab[key]
		else:
			return 0

	def itos(self, index):
		if index < len(self.vocab):
			return self.inverse_vocab[index]
		return "<unk>"

	def convert_sentence_to_ids(self, sentences):
		tokens_to_ids = []
		for sentence in sentences:  
			tokens_to_ids.append(self.convert_tokens_to_ids(sentence))
		return tokens_to_ids

	def convert_tokens_to_ids(self, tokens):
		return [self.stoi(token) for token in tokens]

	def len(self):
		return len(self.vocab)

	def add_word(self, word):
		return self.vocab[word]


if __name__ == "__main__":
	args = parser.parse_args()

	global step
	vocabs = []

	if args.split is True:
		for index, f in enumerate(args.data):
			vocab = Vocab()
			vocab.build_vocab([f])
			vocab.save(args.save_dir + args.vocab_prefix + ".vocab." + str(index) + ".json")
			vocabs.append(vocab)
	else:
		vocab = Vocab()
		vocab.build_vocab(args.data)
		vocab.save(args.save_dir + args.vocab_prefix + ".vocab.json")
		vocabs.append(vocab)

	for index, vocab in enumerate(vocabs):
		print(f'vocabulary size {index+1:d}: {vocab.len():d}')	




