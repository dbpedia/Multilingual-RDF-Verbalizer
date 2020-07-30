from os import listdir
from os.path import isdir
from nltk import word_tokenize

'''
for f in listdir("references"):
    fout = open(f, "w")
    with open("references/" + f, "r") as fin:
        for line in fin:
            text = ' '.join(word_tokenize(line)).lower()
            if text.strip() == "":
                fout.write(" \n")
            else:
                fout.write(text.strip() + "\n")
    fout.close()



'''
ml = []
indexes = []
with open("eval_info", "r") as f:
    for idx, line in enumerate(f):
        parts = line.strip().split(",")
        id = '-'.join(parts[:3])
        if id not in ml:
            ml.append(id)
            indexes.append(idx)

fout = open("hypothesis", "w")
with open("eval_results.txt", "r") as feval:
    for index, line in enumerate(feval):
        if index in indexes:
            fout.write(line.strip() + "\n")
fout.close()

'''
for f in listdir("."):
    if isdir(f):
        fout = open(f + "/hypothesis", "w")
        with open(f + "/eval_results.txt", "r") as feval:
            for index, line in enumerate(feval):
                if index in indexes:
                    fout.write(line.strip() + "\n")
        fout.close()
'''
