import json
import argparse


parser = argparse.ArgumentParser(description="Getting the embeddings for an specific webnlg dataset")
parser.add_argument(
	'-ref','--reference', type=str, required=True, help='Reference file (a json file)')
parser.add_argument(
	'-hyp', '--hypothesis', type=str, required=True, help='Hypothesis file')

def compute_accuracy(gold, hyp_path):

	y_pred = []
	with open(hyp_path) as f:
		y_pred = f.read().split('\n')[:-1]

	y_real = []
	for i, g in enumerate(gold):
		t = [' '.join(target['output']).lower() for target in g['targets']]
		y_real.append(t)

	num, dem = 0.0, 0
	for i, y_ in enumerate(y_pred):
		y = y_real[i]
		if y_.strip() in y:
			num += 1
		dem += 1
	print('Accuracy: ', round(num/dem, 2))

if __name__ == "__main__":
	args = parser.parse_args()
	gold = json.load(open(args.reference))
	compute_accuracy(gold, args.hypothesis)

	'''
	gold_path = "data/ordering/dev.json"
	gold = json.load(open(gold_path))
	#hyp_path = "output/tr.enc.ordering/dev.eval.0.out"
	hyp_path = "output/ordering/dev.eval.0.out"

	compute_accuracy(gold, hyp_path)

	gold_path = "data/ordering/test.json"
	gold = json.load(open(gold_path))
	#hyp_path = "output/tr.enc.ordering/test.eval.0.out"
	hyp_path = "output/ordering/test.eval.0.out"

	compute_accuracy(gold, hyp_path)
	'''
