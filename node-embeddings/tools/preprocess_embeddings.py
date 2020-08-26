import argparse
import pickle
import numpy as np
import json

parser = argparse.ArgumentParser(description="Getting the embeddings for an specific webnlg dataset")
parser.add_argument(
  '-v-in', '--vocab-in-src', type=str, required=False, help=' json file which contains all the vocabulary (nodes) of the knowledge base')
parser.add_argument(
  '-w-in', '--weight-in-src', type=str, required=False, help='numpy array which contains all the node embeddings of the knowledge base')
parser.add_argument(
  '-v-out', '--vocab-out-src', type=str, required=False, help='Json file or pickle (tensor flow) of an specific dataset')
parser.add_argument(
  '-w-out', '--weight-out-src', type=str, required=False, help='Numpy array of our specific vocabulary')


args = parser.parse_args()

if __name__ == '__main__':

    vocab_in = dict()
    with open(args.vocab_in_src, "r") as f:
        vocab_in = json.load(f)

    weight_in = np.load(args.weight_in_src)

    vocab_out = dict()
    with open(args.vocab_out_src, "rb") as f:
        vocab_out = pickle.load(f).word_index

    vocab_in = {w.lower():i for w, i in vocab_in.items()}

    shape = list(weight_in.shape)
    shape[0] = len(vocab_out)

    weight_out = np.random.normal(np.mean(weight_in), np.std(weight_in), tuple(shape))

    for token in vocab_out:
        if token in vocab_in:
            weight_out[vocab_out[token]-1] = weight_in[vocab_in[token]-1]

    np.save(args.weight_out_src, weight_out)
