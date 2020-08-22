

def split_struct(text):
    sentences, triples, triple = [], [], []
    for w in text:
        if w not in ['<SNT>', '</SNT>', '<TRIPLE>', '</TRIPLE>']:
            triple.append(w)
        elif w == '</TRIPLE>':
            triples.append(triple)
            triple = []
        elif w == '</SNT>':
            sentences.append(triples)
            triples = []
    return sentences


dataset = "dev"

out = open(dataset + ".format.trg", "w")
with open(dataset + ".trg", "r") as f:
	lines = [line.strip() for line in f if line.strip() != ""]
	for line in lines:
		#print(line)
		s = split_struct(line.split())
		f_sentence = ""
		for sentence in s:
			f_sentence = f_sentence + ' <SNT> ' + ' '.join([triple[1] for triple in sentence]) + ' </SNT>'
		out.write(f_sentence.strip() + "\n")
out.close()
