from collections import Counter

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


#my_vocab = Vocab()
#my_vocab.build_vocab(["../data/ordering/dev.src", "../data/structing/dev.src"])

#text = "<sos> <TRIPLE> Abilene_Regional_Airport cityServed Abilene,_Texas </TRIPLE> <TRIPLE> Abilene,_Texas isPartOf Texas </TRIPLE> <eos> <pad> <pad> <pad> <pad>"
#tokens = text.lower().split()
#ids = my_vocab.convert_tokens_to_ids(tokens)
#print(ids)
			
			
	




