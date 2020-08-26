import sys


sys.path.append("..")
from benchmark_reader import Benchmark, select_files

from os import listdir

def load_supplementary(folder):
    lexicon_files = [file for file in listdir(folder) if file.endswith("_lexicon.csv")]
    substitute_files = [file for file in listdir(folder) if file.startswith("dbp-substitute")]

    # Reading lexicon files
    lexicon_triples = list()
    for lex_file in lexicon_files:
        with open(folder + "/" + lex_file) as f:
            lexicon_triples += [(line.split(",")[1].strip(), "modified_relation", line.split(",")[0].strip()) 
                                    for line in f if line.split(",")[0].strip() != "" and line.split(",")[1].strip() != ""]

    substitute_triples = list()
    for subs_file in substitute_files:
        with open(folder + "/" + subs_file) as f:
            substitute_triples += [(line.split("|")[1].strip(), "substitute", line.split("|")[0].strip())
                                    for line in f if line.split("|")[0].strip() != "" and line.split("|")[1].strip() != ""]
    return lexicon_triples, substitute_triples


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
                triples.add((triple.s.strip(), triple.p.strip(),
                  triple.o.strip()))

    return list(triples)


noisy_urls = ["http://dbpedia.org/resource/", "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "http://www.w3.org/2002/07/owl#", "http://dbpedia.org/ontology/", "http://xmlns.com/foaf/0.1/", "http://www.w3.org/2000/01/rdf-schema#", "http://purl.org/dc/terms/", "http://dbpedia.org/datatype/"]

def clean_url(url):
    """
        Removes urls to obtain only the node name
    """
    for noisy_url in noisy_urls:
        url = str(url).replace(noisy_url,"").lower()
    return url

