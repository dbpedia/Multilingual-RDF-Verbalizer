# neural-rdf-verbalizer-gsoc-2020
GSoC 2020 

This repository contains the code of the second part of the Google Summer of Code program.

In order to reproduce the experiments, you will need to install:
- pip install torch

The dataset that I used is the provided [here]. You can find the same dataset in the folder [data]


## 1) Multitask Learning

```
python Train.py -train-src data/ordering/train.src data/structing/train.src data/lexicalization/train.src data/end2end/train.src \
	-train-tgt data/ordering/train.trg data/structing/train.trg data/lexicalization/train.bpe.trg data/end2end/train.bpe.trg \
	-dev-src data/ordering/dev.src data/structing/dev.src data/lexicalization/dev.src data/end2end/dev.src \
	-dev-tgt data/ordering/dev.trg data/structing/dev.trg data/lexicalization/dev.bpe.trg data/end2end/dev.bpe.trg \
	-mtl -batch-size 64 -max-length 180 -lr 0.001 -seed 13 -clipping 1 \
	-hidden-size 256 -enc-layers 3 -dec-layers 3 -enc-filter-size 512 -dec-filter-size 512 \
	-enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 \
	-steps 100000 -eval-steps 500 -print-every 100 \
	-eval data/ordering/dev.eval data/structing/dev.eval data/lexicalization/dev.eval data/end2end/dev.eval \
	-test data/ordering/test.eval data/structing/test.eval data/lexicalization/test.eval data/end2end/test.eval \
	-gpu -save-dir outputs/4.all/
```

## 2) Transfer Learning

If you want to run transfer learning experiments, you need to generate a share vocabulary for the source and the target. You can do this running this code:
```
python utils/vocab.py -data data/structing/train.trg data/ordering/train.trg \
	lexicalization/train.bpe.trg data/end2end/train.bpe.trg \
	-vocab-prefix tgt -save-dir vocab/
```

You must run this code for the source. Otherwise, you can use the processed [vocabulary].

Here is an example of how to train on the ordering dataset and then finetune on the structuring dataset using only the decoder trained on the previous step.

```
python Train.py -train-src data/ordering/train.src -train-tgt data/ordering/train.trg \
	-dev-src data/ordering/dev.src -dev-tgt data/ordering/dev.trg \
	-mtl -batch-size 64 -max-length 180 -lr 0.001 -seed 13 -clipping 1 -hidden-size 256 \
	-enc-layers 3 -dec-layers 3 -enc-filter-size 512 -dec-filter-size 512 \
	-enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 \
	-steps 25000 -eval-steps 500 -print-every 100 -gpu \
	-eval data/ordering/dev.eval -test data/ordering/test.eval \
	-save-dir outputs/tr.ordering/ \
	-src-vocab vocab/src.vocab.json -tgt-vocab vocab/tgt.vocab.json
```

The params `model` and `load-encoder` allow to load the pre-trained model (in the discourse ordering task) and get the encoder of that model to train the Text structuring task.

```
python Train.py -train-src data/structing/train.src -train-tgt data/structing/train.trg \
	-dev-src data/structing/dev.src -dev-tgt data/structing/dev.trg \
	-mtl -batch-size 64 -max-length 180 -lr 0.001 -seed 13 -clipping 1 -hidden-size 256 \
	-enc-layers 3 -dec-layers 3 -enc-filter-size 512 -dec-filter-size 512 \
	-enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 \
	-steps 25000 -eval-steps 500 -print-every 100 -gpu \
	-eval data/structing/dev.eval -test data/structing/test.eval \
	-save-dir outputs/tr.structuring/ \
	-src-vocab vocab/src.vocab.json -tgt-vocab vocab/tgt.vocab.json \
	-model outputs/tr.ordering/model.pt -load-encoder
```

[vocabulary]: https://github.com/dbpedia/Multilingual-RDF-Verbalizer/tree/master/pytorch/vocab
[data]: https://github.com/dbpedia/Multilingual-RDF-Verbalizer/tree/master/pytorch/data
[here]: https://github.com/ThiagoCF05/DeepNLG
[Google Sentencepiece]: https://github.com/google/sentencepiece
