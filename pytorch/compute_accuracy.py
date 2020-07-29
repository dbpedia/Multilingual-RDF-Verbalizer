import json

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

print("Ordering")

gold_path = "data/ordering/dev.json"
gold = json.load(open(gold_path))
#hyp_path = "output/tr.enc.ordering/dev.eval.0.out"
hyp_path = "output/4.all/dev.eval.0.out"

compute_accuracy(gold, hyp_path)

gold_path = "data/ordering/test.json"
gold = json.load(open(gold_path))
#hyp_path = "output/tr.enc.ordering/test.eval.0.out"
hyp_path = "output/4.all/test.eval.0.out"

compute_accuracy(gold, hyp_path)

print("Structuring")

gold_path = "data/structing/dev.json"
gold = json.load(open(gold_path))
#hyp_path = "output/tr.enc.structuring/dev.eval.0.out"
hyp_path = "output/4.1.all/dev.eval.1.out"

compute_accuracy(gold, hyp_path)

gold_path = "data/structing/test.json"
gold = json.load(open(gold_path))
#hyp_path = "output/tr.enc.structuring/test.eval.0.out"
hyp_path = "output/4.1.all/test.eval.1.out"

compute_accuracy(gold, hyp_path)

