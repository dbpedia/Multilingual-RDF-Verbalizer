

for fi in ["4.all", "4.1.all"]:
	for name_in, name_out in zip(["dev2.out", "test2.out", "dev3.out", "test3.out"], ["dev2.txt", "test2.txt", "dev3.txt", "test3.txt"]):
		fout = open(fi + "/" + name_out, "w")
		with open(fi + "/" + name_in, "r") as f:
			for line in f:
				tokens = line.split()
				newline = ""
				for token in tokens:
					if token.startswith("▁"):
						newline += " " + token[1:]
					else:
						newline += token
				fout.write(newline.strip() + "\n")
		fout.close()
