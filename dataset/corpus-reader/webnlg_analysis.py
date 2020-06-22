from benchmark_reader import Benchmark
from benchmark_reader import select_files
import json

# where to find the corpus
path_to_corpus = '../challenge2020_train_dev/en/train/'

'''
# initialise Benchmark object
b = Benchmark()

# collect xml files
files = select_files(path_to_corpus)

# load files to Benchmark
b.fill_benchmark(files)
'''

entries = set()

for path_to_corpus in ['../challenge2020_train_dev/en/train/', '../challenge2020_train_dev/en/dev/', '../challenge2020_train_dev/ru/train/', '../challenge2020_train_dev/ru/dev/']:
    b = Benchmark()
    files = select_files(path_to_corpus)
    b.fill_benchmark(files)

    # get access to each entry info
    for entry in b.entries:
        for originaltriple in entry.originaltripleset:
            for triple in originaltriple.triples:
                entries.add(triple.s.lower())
                entries.add(triple.p.lower())
                #entries.add(triple.o)
        
        for triple in entry.modifiedtripleset.triples:
            entries.add(triple.s.lower())
            entries.add(triple.p.lower())
            entries.add(triple.o.lower())

entry_dict = { entry:(i+1) for i, entry in enumerate(entries)}

with open("my-vocab.json", "w", encoding="utf8") as jsonfile:
    json.dump(entry_dict, jsonfile, ensure_ascii=False, indent=4)

#print(entries)
print(len(entries))
