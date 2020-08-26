from benchmark_reader import Benchmark
from benchmark_reader import select_files

import argparse
from os import makedirs
from os import listdir
from os import path

def process_webnlg(filename, output, lang):

    dataset = filename.split("/")[-1]
    print(output + "/" + dataset + "_info")
    print(filename)

    info_file = open(output + "/" + dataset + "_info", "w")
    src_file = open(output + "/" + dataset + "_src", "w")
    tgt_file = open(output + "/" + dataset + "_tgt", "w")

    b = Benchmark()
    files = select_files(filename)
    b.fill_benchmark(files)

    for entry in b.entries:
        #print(entry.id)
        triples = ' <TSP> '.join([triple.s.strip() + " | " + triple.p.strip() + " | " + triple.o.strip() for triple in entry.modifiedtripleset.triples])
        verbalisations = []
        
        for lex in entry.lexs:
            if lex.lang == lang:
                info_file.write(entry.category + "," + entry.size + "," + entry.id + "," + lex.id + "\n")
                src_file.write(triples.strip() + "\n")
                tgt_file.write(lex.lex.strip() + "\n")
        
    info_file.close()
    src_file.close()
    tgt_file.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-webnlg', "--webnlg", required=True, type=str, help="WebNLG dataset folder")
    parser.add_argument('-output', "--output", required=True, type=str, help="Preprocessed WebNLG dataset folder")
    args = parser.parse_args()

    for lang in listdir(args.webnlg):
        if path.isdir(args.webnlg + "/" +lang):
            if not path.exists(args.output + "/" +lang):
                makedirs(args.output + "/" + lang)
            if lang != "ru":
                continue
            for set in listdir(args.webnlg + "/" +lang):
                process_webnlg(args.webnlg + "/" +lang + "/" + set, args.output + "/" + lang, lang)

if __name__ == "__main__":
    main()

