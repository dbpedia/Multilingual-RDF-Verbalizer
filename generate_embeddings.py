import random
import numpy as np
import rdflib
import json
import pickle

from rdf2vec.converters import load_kg
from rdf2vec.walkers import RandomWalker
from rdf2vec.walkers import WeisfeilerLehmanWalker
from rdf2vec import RDF2VecTransformer


import argparse
from os import makedirs

def load_knowledge_graph(filenames, filetype = "nt", strategy="rnd"):
    '''
       Load the knowledge graph for all files
       The parameters strategy optimize to not load all edges in the graph (just directed)
    '''

    # Convert the rdflib to our KnowledgeGraph object
    kg = load_kg(filenames, filetype=filetype, strategy=strategy)
    return kg


def load_vocab(vocab_path):
    '''
        Load the vocabulary according a path.
        The file to be loaded can a pickle (from keras tokenizer) or a json
  
    '''

    if vocab_path.endswith(".json"):
        with open(vocab_path, "r") as f:
            return [w for w in json.load(f).keys()]
    else:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f).word_index
            return [w for w in vocab.keys()]

    return None


def generate_embeddings(kg, vocab, embed_size, walkers=None, n_jobs=1, window_size=5, sg=1, max_iter=10, negative=25, min_count=1):
    '''
       Call the method to train and generate the embeddings
    '''

    transformer = RDF2VecTransformer(vector_size=embed_size, walkers=[walkers], n_jobs= n_jobs, window=window_size, sg=sg, max_iter=max_iter, negative=negative, min_count=min_count)

    walk_embeddings = transformer.fit_transform(kg, vocab)
    embeddings = np.array(walk_embeddings)
    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dbpedia', "--dbpedia", nargs='*', help="DBpedia files")
    parser.add_argument('-vocab', "--vocab", type=str, help="Vocabulary input. It is usually a pickle.")
    parser.add_argument('-embed-size', "--embedding-size", type=int)
    parser.add_argument('-depth', "--depth", type=int)
    parser.add_argument('-algorithm', "--algorithm", type=str, choices=['rnd', 'wl'])
    parser.add_argument('-wl-iter', "--wl-iter", type=int)
    parser.add_argument('-walks', "--walks", type=int, help="Number of walks per instance. If you have to explore all possible paths set this value as -1")
    parser.add_argument('-jobs',"--jobs", default=1, type=int)
    parser.add_argument('-window', "--window-size", type=int, default=5)
    parser.add_argument('-sg', "--skip-gram", action='store_true')
    parser.add_argument('-max-iter',"--max-iter", type=int, default=10)
    parser.add_argument('-negative',"--negative-sampling", type=int, default=25)
    parser.add_argument('-min-count',"--min-count", type=int, default=1)
    parser.add_argument("-ov","--output-vocab", help="DBpedia vocabulary file (json format)")
    parser.add_argument("-oe","--output-embeddings", help="DBpedia embeddings folder")
    parser.add_argument("-seed","--seed", type=int, help="Seed")
    parser.add_argument("-error-file","--error-file", help="This file contains nodes that are not part of the dbpedia files")
    args = parser.parse_args()

    print("Loading vocabulary...")
    vocab_dataset = load_vocab(args.vocab)

    print("Vocabulary size:\t", len(vocab_dataset))
    print("Vocabulary loaded")

    sg = 1 if args.skip_gram else 0

    print("Loading the knowledge graph by using ", model, "...")
    kg = load_knowledge_graph(args.dbpedia, strategy=args.algorithm)
    print("Knowledge Graph loaded")

    if args.walks == -1:
        walk = float('inf')
        folder = args.output_embeddings + "/" + args.algorithm + "/" + str(args.depth) + "/inf"
    else:
        walk = args.walks                    
        folder = args.output_embeddings + "/" + args.algorithm + "/" + str(args.depth) + "/" + str(walk)

    try:    
        makedirs(folder)
    except:
        print(folder + " exists...")

    if args.algorithm == "rnd":
        walker = RandomWalker(args.depth, walk, args.error_file)
    else:
        walker = WeisfeilerLehmanWalker(args.depth, args.walks, args.wl_iter, args.error_file)

    np.random.seed(args.seed)
    random.seed(args.seed)

    embeddings = generate_embeddings(kg, vocab_dataset, args.embedding_size, walkers=walker, n_jobs=args.jobs, window_size=args.window_size, sg=sg, max_iter= args.max_iter, negative=args.negative_sampling, min_count=args.min_count)

    if sg == 1:
        name = folder + "/embeddings." + "sg" + ".emb" + str(args.embedding_size) + ".win" + str(args.window_size)
    else:
        name = folder + "/embeddings." + "cbow" + ".emb" + str(args.embedding_size) + ".win" + str(args.window_size)
        
    np.save(name, embeddings)


if __name__ == "__main__":
    main()
