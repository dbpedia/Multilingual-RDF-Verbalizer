import json

gold_path = "data/structing/test.json"
gold = json.load(open(gold_path))

y_pred = []
hyp_path = "deepnlg/all/test.eval.1.out"
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
