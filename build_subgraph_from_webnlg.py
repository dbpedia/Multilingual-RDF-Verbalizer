from tqdm import tqdm
import rdflib

import argparse
from benchmark_reader import Benchmark
from benchmark_reader import select_files
from rdflib import URIRef, Literal

noisy_urls = ["http://dbpedia.org/resource/", "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "http://www.w3.org/2002/07/owl#", "http://dbpedia.org/ontology/", "http://xmlns.com/foaf/0.1/", "http://www.w3.org/2000/01/rdf-schema#", "http://purl.org/dc/terms/", "http://dbpedia.org/datatype/"]

def clean_uri(url):
    """
        Removes urls to obtain only the node name
    """
    for noisy_url in noisy_urls:
        url = str(url).replace(noisy_url,"").lower()
    return url

def load_knowledgegraph(files, format="nt"):
    hash_graph = dict()
    print("Loading RDF graphs...")
    for f in tqdm(files):
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

    graph = load_knowledgegraph(files,format)

    out = rdflib.Graph()
    nodes = list(vocab)
    iters = 0
    while len(nodes) > 0 and iters < max_depth:
        print("Number of nodes to analyse: ", len(nodes))
        children = set()
        for subject in tqdm(list(nodes)):
            if not isinstance(subject, URIRef):
                subject = URIRef("http://dbpedia.org/resource/"+ subject)
            if subject in graph:
                for _predicate, _object in graph[subject]:
                    if isinstance(_object, URIRef):
                        children.add(_object)
                        out.add((subject,_predicate,_object))
        nodes = list(children)
        iters += 1
    out.serialize(destination=output, format=format)


def build_vocab(files):

    vocab = set()

    for path_to_corpus in files:
        b = Benchmark()
        files = select_files(path_to_corpus)
        b.fill_benchmark(files)

        for entry in b.entries:
            for tripleset in entry.originaltripleset:
                for triple in tripleset.triples:
                    vocab.add(triple.s.strip())
                    #vocab.add(triple.o.lower())
            for triple in entry.modifiedtripleset.triples:
                vocab.add(triple.s.strip())
                #vocab.add(triple.o.lower())

    return vocab

#['challenge2020_train_dev/en/train/', 'challenge2020_train_dev/en/dev/', 'challenge2020_train_dev/ru/train/', 'challenge2020_train_dev/ru/dev/']
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


