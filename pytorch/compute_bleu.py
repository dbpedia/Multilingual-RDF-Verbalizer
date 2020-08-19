import json
import argparse
import subprocess
import os

unseen_domains = ['Artist', 'Politician', 'CelestialBody', 'Athlete', 'MeanOfTransportation']

parser = argparse.ArgumentParser(description="Getting the embeddings for an specific webnlg dataset")
parser.add_argument(
	'-ref','--reference', type=str, required=True, help='Reference file (a json file)')
parser.add_argument(
	'-hyp', '--hypothesis', type=str, required=True, help='Hypothesis file')
parser.add_argument(
	'-type', '--type', type=str, required=True, help='Type: all, seen, unseen')

def compute_bleu(gold, hyp_path, domain):

	_y_pred = []
	with open(hyp_path) as f:
		_y_pred = f.read().split('\n')[:-1]

	y_real = []
	y_pred = []

	y_real, y_pred = [], []
	if domain == "all":
		for i, g in enumerate(gold):
			t = [' '.join(target['output']) for target in g['targets']]
			y_real.append(t)
			y_pred.append(_y_pred[i].strip())
	elif domain == "unseen":
		for i, g in enumerate(gold):
			if g['category'] in unseen_domains:
				t = [' '.join(target['output']) for target in g['targets']]
				y_real.append(t)
				y_pred.append(_y_pred[i].strip())
	else:
		for i, g in enumerate(gold):
			if g['category'] not in unseen_domains:
				t = [' '.join(target['output']) for target in g['targets']]
				y_real.append(t)
				y_pred.append(_y_pred[i].strip())


	with open('predictions', 'w') as f:
		f.write('\n'.join(y_pred))

	nfiles = max([len(refs) for refs in y_real])
	for i in range(nfiles):
		with open('reference' + str(i+1), 'w') as f:
			for refs in y_real:
				if i < len(refs):
					f.write(refs[i])
				f.write('\n')

	bleu_command = 'utils/multi-bleu-detok.perl'
	command = 'perl ' + bleu_command + ' reference1 reference2 reference3 reference4 reference5 reference6 reference7 reference8 < predictions'
	result = subprocess.check_output(command, shell=True)
	print(result.strip())
	try:
		os.remove('reference1')
		os.remove('reference2')
		os.remove('reference3')
		os.remove('reference4')
		os.remove('reference5')
		os.remove('reference6')
		os.remove('reference7')
		os.remove('reference8')
		os.remove('predictions')
	except:
		pass


if __name__ == "__main__":
	args = parser.parse_args()
	gold = json.load(open(args.reference))
	compute_bleu(gold, args.hypothesis, args.type)

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
