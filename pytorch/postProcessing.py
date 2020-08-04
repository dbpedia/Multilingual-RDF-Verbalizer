

path = "output/lexicalization"
for name_in, name_out in zip(["dev.eval.0.out", "test.eval.0.out"], ["dev.lower.out", "test.lower.out"]):
	fout = open(path + "/" + name_out, "w")
	with open(path + "/" + name_in, "r") as f:
		for line in f:
			tokens = line.split()
			newline = ""
			for token in tokens:
				if token.endswith("@@"):
					newline += token.replace("@@","")
				else:
					newline += token + " "
			fout.write(newline.strip() + "\n")
	fout.close()
