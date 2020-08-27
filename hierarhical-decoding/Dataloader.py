from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils.vocab import Vocab
import torch

class ParallelDataset(Dataset):
	'''
		This class builds a dataset from source/target files according to a max_length
	'''

	def __init__(self, source_name, target_name, max_length=300, source_vocab=None, target_vocab=None):

		self.data_source = self.read_file(source_name)
		self.data_target = self.read_file(target_name)

		self.max_length = max_length

		self.source_vocab = source_vocab
		if source_vocab == None:
			self.source_vocab = Vocab()
			self.source_vocab.build_vocab([source_name])

		self.target_vocab = target_vocab
		if target_vocab == None:
			self.target_vocab = Vocab()
			self.target_vocab.build_vocab([target_name])

			
	def __len__(self):
		'''
			Return the length of the dataset
		'''
		return len(self.data_source)

	def __getitem__(self, index):

		src_tokens = self.padding_sentence(self.data_source[index])
		tgt_tokens = self.padding_sentence(self.data_target[index])

		src_tokens_ids = self.source_vocab.convert_tokens_to_ids(src_tokens)
		src_tokens_ids_tensor = torch.tensor(src_tokens_ids)

		tgt_tokens_ids = self.target_vocab.convert_tokens_to_ids(tgt_tokens)
		tgt_tokens_ids_tensor = torch.tensor(tgt_tokens_ids)


		return src_tokens_ids_tensor, tgt_tokens_ids_tensor


	def read_file(self, filename):
		'''
			Read the file to 
			filename: filename or path of the source/target files
		'''
		data = []
		with open(filename, "r") as f:
			for line in f:
				data.append(line.strip().split()) 
		return data


	def padding_sentence(self, tokens):
		'''
			Padding the sentence (adding sos and eos tokens and fix the length to a max_length
			tokens: list of tokens of a sentence
		'''
		tokens = ['<sos>'] + tokens + ['<eos>']

		if len(tokens) < self.max_length:
			tokens = tokens + ['<pad>' for _ in range(self.max_length - len(tokens))]
		else:
			tokens = tokens[:self.max_length-1] + ['<eos>']

		return tokens


	def vocabs(self):
		'''
			Return the source and target vocabulary
		'''
		return self.source_vocab, self.target_vocab


def get_dataloader (dataset, batch_size, shuffle=False):
	'''
		return a DataLoader object
		dataset: the parallel dataset
		batch_size: 
		shuffle: if we should shuffle the dataset
	'''
	return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 1)

