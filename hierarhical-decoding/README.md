# Hierachical Decoding - GSoC 2020 

This folder contains the code of the second part of the Google Summer of Code program.

In order to reproduce the experiments, you will need to install the `requirements.txt`. Additionally, you will need to install [Moses] to run the preprocessing steps and download [Subword-NMT] to get Byte-pair encoding in the target files.
```
	git clone https://github.com/rsennrich/subword-nmt.git
```

The dataset that I used is the provided [here]. You can find the same dataset in the folder [data]

## 1) Preprocessing files

The preprocessing step is only used for lexicalisation and End2End tasks

```
./preprocess.sh data/en/end2end case_model bpe_model
```

You should verify if the path of Moses and Subword-NMT in the file in correct. This bash will process the train.trg file and output a file called train.bpe.trg in the same `data/en/end2end` folder.


For Multitask and Transfer learning we need to pass a pre-generated shared vocabulary. This vocabulary (`tied.vocab.json`) can be found in the folder [vocab]. If you want to generate one, you should run this script (this is an example):

```
python utils/vocab.py -data data/en/ordering/train.src data/en/structing/train.src data/en/ordering/train.trg data/en/structing/train.trg \
	-vocab-prefix tied -save-dir vocab/
```

It is worth noting that the folder `data` already contains all the files processed. 


## 2) Training

### 2.1) Multitask Learning

If you want to train the Multitask learning approach, you should run this script

```
python Train.py -train-src data/en/ordering/train.src data/en/structing/train.src data/en/lexicalization/train.src data/en/end2end/train.src \
	-train-tgt data/en/ordering/train.trg data/en/structing/train.trg data/en/lexicalization/train.bpe.trg data/en/end2end/train.bpe.trg \
	-dev-src data/en/ordering/dev.src data/en/structing/dev.src data/en/lexicalization/dev.src data/en/end2end/dev.src \
	-dev-tgt data/en/ordering/dev.trg data/en/structing/dev.trg data/en/lexicalization/dev.bpe.trg data/en/end2end/dev.bpe.trg \
    -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 \
    -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 \
    -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 \
    -enc-dropout 0.1 -dec-dropout 0.1 \
    -steps 800000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 \
	-eval data/en/ordering/dev.eval data/en/structing/dev.eval data/en/lexicalization/dev.eval data/en/end2end/dev.eval \
	-test data/en/ordering/test.eval data/en/structing/test.eval data/en/lexicalization/test.eval data/en/end2end/test.eval \
	-gpu -save-dir output/mtl.4.4/ -tie-embeddings -src-vocab vocab/tied.vocab.json -beam-size 5
```

### 2.2) Transfer Learning

Here is an example of how to train on the ordering dataset and then finetune on the structuring dataset using only the decoder trained on the previous step.

```
python Train.py -train-src data/ordering/train.src -train-tgt data/ordering/train.trg \
	-dev-src data/ordering/dev.src -dev-tgt data/ordering/dev.trg \
    -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 \
    -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 \
    -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 \
    -enc-dropout 0.1 -dec-dropout 0.1 -gpu \
    -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 \
	-eval data/ordering/dev.eval -test data/ordering/test.eval \
	-save-dir output/tr.ordering/ \
	-tie-embeddings -src-vocab vocab/tied.vocab.json -beam-size 5
```

The params `model` and `load-encoder` allow to load the pre-trained model (in the discourse ordering task) and get the encoder of that model to train the Text structuring task. If you want share all the model (not only the encoder) do not put `load-encoder`.

```
python Train.py -train-src data/structing/train.src -train-tgt data/structing/train.trg \
	-dev-src data/structing/dev.src -dev-tgt data/structing/dev.trg \
    -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 \
    -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 \
    -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 \
    -enc-dropout 0.1 -dec-dropout 0.1 -gpu \
    -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 \
	-eval data/structing/dev.eval -test data/structing/test.eval \
	-save-dir output/tr.structuring/ \
	-model output/tr.ordering/model.pt -load-encoder
	-tie-embeddings -src-vocab vocab/tied.vocab.json -beam-size 5
```

### 2.3) Multi-input (similar to Multilingual NMT)
To run the multi-input approach you need to add a task token in each source file. The multi `data/en/multi` already contained all the modifications. However, You can preprocess all the source files by running the `multi_preprocess.sh`. This file considers that lexicalisation and End2End tasks already have been preprocessed together (you can see the files in the `data/en/end2end+lexicalization` folder).

```
python Train.py -train-src data/en/multi/train.src -train-tgt data/en/multi/train.trg \
	-dev-src data/en/multi/dev.src -dev-tgt data/en/multi/dev.trg \
    -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 \
    -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 \
    -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 \
    -enc-dropout 0.1 -dec-dropout 0.1 -gpu \
    -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 \
	-eval data/en/multi/dev.eval -test data/en/multi/test.eval \
	-save-dir output/multi/ \
	-model output/tr.ordering/model.pt -load-encoder
	-tie-embeddings -beam-size 5
```

If you already have a model and only want to execute. You should add the param `-translate` in the train.py.

## 3) Postprocessing

To postprocessing the outputs of the lexicalisation and the End2End task, you need to run this script:
```
./postprocess.sh output/end2end/dev.eval.0.out data/en/end2end/case_model en
```

## 4) Pipeline

Finally, you can run all the models hierarchically, i.e., from the high-level tasks to the low-level tasks. In particular, this script run the Discourse Ordering, Text Structuring, Lexicalisation, Referring Expression generation and Lexicalisation tasks sequentially. You will need to train the [NeuralREG] tool or use the model available in this [link].

```
./pipeline.sh
```
In this bash script you should change the variables `pipeline_dir` (putting your pipeline output folder), `model_dir` (the path where all models are located), and `neuralreg_dir` (the path where the neuralreg model is). Additionally, you need to change the variables `project_dir`, `moses` and `lng`.


[data]: https://github.com/dbpedia/Multilingual-RDF-Verbalizer/tree/master/hierarhical-decoding/data
[here]: https://github.com/ThiagoCF05/DeepNLG
[Moses]: https://github.com/moses-smt/mosesdecoder.git
[Subword-NMT]: https://github.com/rsennrich/subword-nmt.git
[vocab]: https://github.com/dbpedia/Multilingual-RDF-Verbalizer/tree/master/hierarhical-decoding/vocab
[NeuralREG]: https://github.com/ThiagoCF05/NeuralREG
[link]: https://drive.google.com/drive/folders/13GPCKtAtI2y_fzNVWAQ_9Ccb-H2TRu00?usp=sharing
