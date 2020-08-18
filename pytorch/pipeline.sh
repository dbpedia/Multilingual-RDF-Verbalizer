
#for set in dev test
#do

python /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/Train.py -train-src /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/ordering/train.src -train-tgt /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/ordering/train.trg -dev-src /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/ordering/dev.src -dev-tgt /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/ordering/dev.trg -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 -clipping 1 -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 -gpu -eval /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/ordering/dev.eval -test /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/ordering/test.eval -save-dir /home/marcosbc/results/output.03082020/ordering/ -tie-embeddings -model -translate -beam-size 5

mv /home/marcosbc/results/output.03082020/ordering/dev.eval.0.out /home/marcosbc/results/pipeline/ordering.dev
mv /home/marcosbc/results/output.03082020/ordering/test.eval.0.out /home/marcosbc/results/pipeline/ordering.test

python /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/mapping.py /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/ordering/dev.eval /home/marcosbc/results/pipeline/ordering.dev ordering /home/marcosbc/results/pipeline/ordering.mapped.dev

python /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/mapping.py /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/ordering/test.eval /home/marcosbc/results/pipeline/ordering.test ordering /home/marcosbc/results/pipeline/ordering.mapped.test


#done
