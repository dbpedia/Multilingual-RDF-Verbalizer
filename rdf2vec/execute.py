import random
import numpy as np
import rdflib
import json
import pickle
import tqdm
import sys

from converters import load_kg
from walkers import RandomWalker
from walkers import WeisfeilerLehmanWalker
from _rdf2vec import RDF2VecTransformer
from graph import KnowledgeGraph, Vertex

import argparse
from os import makedirs
from os import listdir

sys.path.append("..")
from benchmark_reader import Benchmark
from benchmark_reader import select_files

def load_supplementary(folder):
    """
        Load supplementary information about modified relations and properties in the webnlg dataset.
    """

    lexicon_files = [file for file in listdir(folder) if file.endswith("_lexicon.csv")]
    substitute_files = [file for file in listdir(folder) if file.startswith("dbp-substitute")]

    # Reading lexicon files
    lexicon_triples = list()
    for lex_file in lexicon_files:
        with open(folder + "/" + lex_file) as f:
            lexicon_triples += [(line.split(",")[1].strip().lower(), "modified_relation", line.split(",")[0].strip().lower()) 
                                    for line in f if line.split(",")[0].strip() != "" and line.split(",")[1].strip() != ""]

    substitute_triples = list()
    for subs_file in substitute_files:
        with open(folder + "/" + subs_file) as f:
            substitute_triples += [(line.split("|")[1].strip().lower(), "substitute", line.split("|")[0].strip().lower())
                                    for line in f if line.split("|")[0].strip() != "" and line.split("|")[1].strip() != ""]
    return lexicon_triples, substitute_triples

def add_relations(kg, triples):
    """
        Add relations to the current Knowledge graph
        kg: Knowledge graph
        triples: new triples to add (s,p,o) format
    """
    for (s, p, o) in triples:
        
        s_v = Vertex(str(s))
        o_v = Vertex(str(o))
        p_v = Vertex(str(p), predicate=True, _from=s_v, _to=o_v)
        kg.add_vertex(s_v)
        kg.add_vertex(p_v)
        kg.add_vertex(o_v)
        kg.add_edge(s_v, p_v)
        kg.add_edge(p_v, o_v)


def load_knowledge_graph(filenames, filetype = "nt", strategy="rnd"):
    '''
       Load the knowledge graph for all files
       The parameters strategy optimize to not load all edges in the graph (just directed)
       filenames: file names of all .ttl (knowledge base)
       filetype: rdf file type. This is used in the RDFlib parser
       strategy: "rnd" or "wl". Inverse relations are loaded as well if "wl" is used
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

def load_triples_webnlg(files):
    """
       Load all triples of the webnlg files 
       files: webnlg files (train, dev for each language)
    """

    triples = set()
    for path_to_corpus in files:
        b = Benchmark()
        files = select_files(path_to_corpus)
        b.fill_benchmark(files)

        for entry in b.entries:
            for triple in entry.modifiedtripleset.triples:
                triples.add((triple.s.strip().lower(), triple.p.strip().lower(),
                  triple.o.strip().lower()))

    return list(triples)



def generate_embeddings(kg, vocab, embed_size, walkers=None, n_jobs=1, window_size=5, sg=1, max_iter=10, negative=25, min_count=1, print_walks=False):
    '''
       Call the method to train and generate the embeddings
    '''

    transformer = RDF2VecTransformer(vector_size=embed_size, walkers=[walkers], n_jobs= n_jobs, window=window_size, sg=sg, max_iter=max_iter, negative=negative, min_count=min_count, print_walks= print_walks)

    walk_embeddings = transformer.fit_transform(kg, vocab)
    embeddings = np.array(walk_embeddings)
    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dbpedia', "--dbpedia", nargs='*', help="DBpedia files")
    parser.add_argument('-vocab', "--vocab", type=str, help="Vocabulary input. It is usually a pickle.")
    parser.add_argument('-webnlg', "--webnlg", nargs='*', required=False, help="WebNLG dataset (en/train en/dev ...)")
    parser.add_argument('-sup', "--supplementary", type=str, required=False, help="Folder where there are several triple modifications")
    parser.add_argument('-embed-size', "--embedding-size", type=int)
    parser.add_argument('-depth', "--depth", type=int)
    parser.add_argument('-algorithm', "--algorithm", type=str, default="rnd", choices=['rnd', 'wl'])
    parser.add_argument('-wl-iter', "--wl-iter", type=int)
    parser.add_argument('-walks', "--walks", type=int, help="Number of walks per instance. If you have to explore all possible paths set this value as -1")
    parser.add_argument('-jobs',"--jobs", default=1, type=int)
    parser.add_argument('-window', "--window-size", type=int, default=5, help="window size of the word2vec algorithm (default: 5)")
    parser.add_argument('-sg', "--skip-gram", action='store_true', help="If we are use skip-gram (default cbow)")
    parser.add_argument('-max-iter',"--max-iter", type=int, default=10)
    parser.add_argument('-negative',"--negative-sampling", type=int, default=25)
    parser.add_argument('-min-count',"--min-count", type=int, default=1)
    parser.add_argument("-oe","--output-embeddings", help="DBpedia embeddings folder")
    parser.add_argument("-seed","--seed", type=int, help="Seed")
    parser.add_argument("-print-walks","--print-walks", action='store_true', help="Parameter to print all paths. This makes slower all process")
    args = parser.parse_args()

    print("Loading vocabulary...")
    vocab_dataset = load_vocab(args.vocab)
    print("Vocabulary loaded. Vocabulary size:", len(vocab_dataset))
    
    sg = 1 if args.skip_gram else 0

    print("Loading the knowledge graph by using", args.algorithm, "...")
    kg = load_knowledge_graph(args.dbpedia, strategy=args.algorithm)
    print("Knowledge Graph loaded. Number of vertices:", len(kg._vertices))

    if args.supplementary is not None:
        print("Loading supplementary data ...")
        lexicon_triples, substitute_triples = load_supplementary(args.supplementary)
        print("Supplementary data loaded. (" + str(len(lexicon_triples)), "lexicon triples and", 
          str(len(substitute_triples)), "substitute triples)")
        add_relations(kg, lexicon_triples)
        add_relations(kg, substitute_triples)
        print("Knowledge Graph updated. Number of vertices after supplementary data:", len(kg._vertices))

    if args.webnlg is not None:
        print("Loading WebNLG triples ...")
        webnlg_triples = load_triples_webnlg(args.webnlg)
        print(str(len(webnlg_triples)), "modified WebNLG triples loaded")
        add_relations(kg, webnlg_triples)
        print("Knowledge Graph updated. Number of vertices after webnlg triples:", len(kg._vertices))


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
        walker = RandomWalker(args.depth, walk)
    else:
        walker = WeisfeilerLehmanWalker(args.depth, args.walks, args.wl_iter)

    np.random.seed(args.seed)
    random.seed(args.seed)
        

    embeddings = generate_embeddings(kg, vocab_dataset, args.embedding_size, walkers=walker, n_jobs=args.jobs, window_size=args.window_size, sg=sg, max_iter= args.max_iter, negative=args.negative_sampling, min_count=args.min_count, print_walks=args.print_walks)

    if sg == 1:
        name = folder + "/embeddings." + "sg" + ".emb" + str(args.embedding_size) + ".win" + str(args.window_size)
    else:
        name = folder + "/embeddings." + "cbow" + ".emb" + str(args.embedding_size) + ".win" + str(args.window_size)

    np.save(name, embeddings)
    

if __name__ == "__main__":
    main()
