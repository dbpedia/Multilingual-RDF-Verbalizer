import json
import argparse

unseen_domains = ['Artist', 'Politician', 'CelestialBody', 'Athlete', 'MeanOfTransportation']

parser = argparse.ArgumentParser(description="Getting the embeddings for an specific webnlg dataset")
parser.add_argument(
	'-ref','--reference', type=str, required=True, help='Reference file (a json file)')
parser.add_argument(
	'-hyp', '--hypothesis', type=str, required=True, help='Hypothesis file')
parser.add_argument(
	'-type', '--type', type=str, required=True, help='Type: all, seen, unseen')

def compute_accuracy(gold, hyp_path, domain):

	_y_pred = []
	with open(hyp_path) as f:
		_y_pred = f.read().split('\n')[:-1]

	y_real = []
	y_pred = []
	if domain == "all":
		for i, g in enumerate(gold):
			t = [' '.join(target['output']).lower() for target in g['targets']]
			y_real.append(t)
			y_pred.append(_y_pred[i])
	elif domain == "unseen":
		for i, g in enumerate(gold):
			if g['category'] in unseen_domains:
				t = [' '.join(target['output']).lower() for target in g['targets']]
				y_real.append(t)
				y_pred.append(_y_pred[i])
	else:
		for i, g in enumerate(gold):
			if g['category'] not in unseen_domains:
				t = [' '.join(target['output']).lower() for target in g['targets']]
				y_real.append(t)
				y_pred.append(_y_pred[i])

	num, den = 0.0, 0
	for i, y_ in enumerate(y_pred):
		y = y_real[i]
		if y_.strip() in y:
			num += 1
		den += 1
	if den == 0:
		print('Accuracy: 0')
	else:
		print('Accuracy: ', round(num/den, 2))

if __name__ == "__main__":
	args = parser.parse_args()
	gold = json.load(open(args.reference))
	compute_accuracy(gold, args.hypothesis, args.type)