import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="preprocessor parser")
parser.add_argument(
  '-csv', '--csv', type=str, required=True, help='Pyke file (usually a .csv)')
parser.add_argument(
  '-embed-size', '--embedding-size', type=int, required=True, help='Embedding size in the csv file')

args = parser.parse_args()

if __name__ == '__main__':

    vocab = dict()
    embeddings = []
    name = args.csv.replace(".csv", "")

    with open(args.csv, "r") as f:
        for idx, line in enumerate(f):
            parts = line.split(",")
            start = len(parts) - args.embedding_size
            id = ','.join(parts[:start])
            vocab[id] = idx + 1
            embeddings.append([float(e) for e in parts[start:]])

    with open(name  +'.json', 'w', encoding="utf-8") as outfile:
        json.dump(vocab, outfile, ensure_ascii=False)

    np.save(name + ".npy", np.asarray(embeddings))

