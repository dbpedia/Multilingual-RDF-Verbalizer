from tqdm import tqdm
import rdflib
import sys

import argparse
from rdflib import URIRef, Literal

sys.path.append("..")
from benchmark_reader import Benchmark
from benchmark_reader import select_files


def load_knowledgegraph(files, format="nt"):
    """
       Load knowledge graph using dbpedia files
       files: usually ttl's which contains triples
       format: format of the files (default nt)

    """

    hash_graph = dict()
    print("Loading RDF graphs...")
    for f in files:
        print("Reading ", f)
        g = rdflib.Graph()
        g.parse(f, format=format)
        for _subject, _predicate, _object in g:
            predicate_objects = []
            if _subject in hash_graph:
                predicate_objects = hash_graph[_subject]
            predicate_objects.append((_predicate, _object))
            hash_graph[_subject] = predicate_objects
        del g
    print("RDF graphs loaded")
    print("Number of keys: ", len(hash_graph))
    return hash_graph


def build_rdf_subgraph(files, vocab, max_depth, output, format="nt"):
    """
       Build a RDF subgraph using a specific vocabulary and all dbpedia files
       files: dbpedia files (containing triples)
       vocab: list of subjects
       max_depth: max depth to search in the dbpedia files
       output: file where subgraphs will be stored
       format: format of the dbpedia files (default: nt)
    """


    graph = load_knowledgegraph(files,format)

    out = rdflib.Graph()
    nodes = list(vocab)
    iters = 0
    while len(nodes) > 0 and iters < max_depth:
        print("Number of nodes to analyse: ", len(nodes))
        not_found = 0
        children = set()
        for subject in list(nodes):
            if not isinstance(subject, URIRef):
                subject = URIRef("http://dbpedia.org/resource/"+ subject)
            if subject in graph:
                for _predicate, _object in graph[subject]:
                    if isinstance(_object, URIRef):
                        children.add(_object)
                    out.add((subject,_predicate,_object))
            else:
                not_found += 1
        print("Nodes not found " +  str(not_found) + "/" + str(len(nodes)))
        nodes = list(children)
        iters += 1
    fout = open(output, "w")
    for (s, p, o) in out:
        fout.write(s.n3() + " " + p.n3() + " " + o.n3() + " .\n")
    fout.close()


def build_vocab(files):
    """
       Build a vocabulary from webnlg dataset. This method only considers the subjects
       files: files (or folders) of the WebNLG dataset
    """

    vocab = set()

    for path_to_corpus in files:
        b = Benchmark()
        files = select_files(path_to_corpus)
        b.fill_benchmark(files)

        for entry in b.entries:
            for tripleset in entry.originaltripleset:
                for triple in tripleset.triples:
                    vocab.add(triple.s.strip())
            for triple in entry.modifiedtripleset.triples:
                vocab.add(triple.s.strip())

    return vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-webnlg', "--webnlg", nargs='*', required=True)
    parser.add_argument('-dbpedia', "--dbpedia", nargs='*', required=True)
    parser.add_argument('-max-depth', "--max-depth", type=int)
    parser.add_argument('-output', "--output", type=str, required=True)
    args = parser.parse_args()
    vocab = build_vocab(args.webnlg)

    if args.max_depth is not None:
        print("Setting max depth: ", args.max_depth)
        build_rdf_subgraph(args.dbpedia, vocab, args.max_depth, args.output)
    else:
        print("No max depth parameter found")
        print("Setting max depth (default): 80")
        build_rdf_subgraph(args.dbpedia, vocab, 80, args.output)

if __name__ == "__main__":
    main()


