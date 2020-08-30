import sys

import argparse

parser = argparse.ArgumentParser(description="Post processing")
parser.add_argument(
	'-i','--input', type=str, required=True, help='Input file')
parser.add_argument(
	'-o', '--output', type=str, required=True, help='Output file')

if __name__ == "__main__":
	args = parser.parse_args()
	fout = open(args.output, "w")
	with open(args.input, "r") as f:
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
