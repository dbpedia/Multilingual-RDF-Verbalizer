
prefix = "dev"

refs = open("data/multibpe/"+ prefix +".eval", "r")
hyps = open("output/multibpe/" + prefix + ".multibpe.0.out", "r") 

keys = {"<ordering>":0, "<structuring>":1, "<lexicalization>":2, "<end2end>":3}

out0 = open("output/multibpe/"+ prefix +".eval.0.out", "w")
out1 = open("output/multibpe/"+ prefix +".eval.1.out", "w")
out2 = open("output/multibpe/"+ prefix +".eval.2.out", "w")
out3 = open("output/multibpe/"+ prefix +".eval.3.out", "w")


for ref, hyp in zip(refs, hyps):
	task = ref.strip().split()[0]
	ref = ' '.join(ref.strip().split()[1:])
	if keys[task.strip()] == 0:
		out0.write(hyp.strip()+"\n")

	if keys[task.strip()] == 1:
		out1.write(hyp.strip()+"\n")

	if keys[task.strip()] == 2:
		out2.write(hyp.strip()+"\n")

	if keys[task.strip()] == 3:
		out3.write(hyp.strip()+"\n")

out0.close()
out1.close()
out2.close()
out3.close()
