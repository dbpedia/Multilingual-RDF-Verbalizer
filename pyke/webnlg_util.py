from benchmark_reader import Benchmark
from benchmark_reader import select_files
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
