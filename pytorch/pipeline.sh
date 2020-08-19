
moses="/home/marcosbc/LinearAMR/mosesdecoder"
lng="en"

#python /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/Train.py -train-src /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/ordering/train.src -train-tgt /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/ordering/train.trg -dev-src /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/ordering/dev.src -dev-tgt /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/ordering/dev.trg -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 -gpu -eval /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/end2end/dev.eval -test /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/end2end/test.eval -save-dir /home/marcosbc/results/output.03082020/ordering/ -tie-embeddings -translate -beam-size 5

#for set in dev test
#do
#mv /home/marcosbc/results/output.03082020/ordering/$set.eval.0.out /home/marcosbc/results/pipeline/ordering.$set

#python /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/mapping.py /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/end2end/$set.eval /home/marcosbc/results/pipeline/ordering.$set ordering /home/marcosbc/results/pipeline/ordering.mapped.$set
#done

#python /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/Train.py -train-src /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/structing/train.src -train-tgt /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/structing/train.trg -dev-src /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/structing/dev.src -dev-tgt /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/structing/dev.trg -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 -gpu -eval /home/marcosbc/results/pipeline/ordering.mapped.dev -test /home/marcosbc/results/pipeline/ordering.mapped.test -save-dir /home/marcosbc/results/output.03082020/structuring/ -tie-embeddings -translate -beam-size 5

#for set in dev test
#do
#mv /home/marcosbc/results/output.03082020/structuring/ordering.mapped.$set.0.out /home/marcosbc/results/pipeline/structuring.$set

#python /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/mapping.py /home/marcosbc/results/pipeline/ordering.mapped.$set /home/marcosbc/results/pipeline/structuring.$set structing /home/marcosbc/results/pipeline/structuring.mapped.$set
#done

python /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/Train.py -train-src /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/lexicalization/train.src -train-tgt /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/lexicalization/train.bpe.trg -dev-src /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/lexicalization/dev.src -dev-tgt /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/lexicalization/dev.bpe.trg -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 -gpu -eval /home/marcosbc/results/pipeline/structuring.mapped.dev -test /home/marcosbc/results/pipeline/structuring.mapped.test -save-dir /home/marcosbc/results/output.03082020/lexicalization/ -tie-embeddings -translate -beam-size 5

for set in dev test
do
mv /home/marcosbc/results/output.03082020/lexicalization/structuring.mapped.$set.0.out /home/marcosbc/results/pipeline/lex.$set

python /home/marcosbc/Projects/gsoc/Multilingual-RDF-Verbalizer/pytorch/postProcessing.py -i /home/marcosbc/results/pipeline/lex.$set -o /home/marcosbc/results/pipeline/lexicalization.lower.$set

$moses/scripts/recaser/recase.perl --in /home/marcosbc/results/pipeline/lexicalization.lower.$set --model /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/data/en/lexicalization/case_model/moses.ini --moses $moses/bin/moses > /home/marcosbc/results/pipeline/lexicalization.cs.$set

$moses/scripts/tokenizer/normalize-punctuation.perl -l $lng < /home/marcosbc/results/pipeline/lexicalization.cs.$set > /home/marcosbc/results/pipeline/lexicalization.punc.$set
$moses/scripts/tokenizer/detokenizer.perl -l $lng < /home/marcosbc/results/pipeline/lexicalization.punc.$set > /home/marcosbc/results/pipeline/lexicalization.$set

python /home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/utils/generate.py /home/marcosbc/results/pipeline/lexicalization.$set /home/marcosbc/results/pipeline/ordering.mapped.$set /home/marcosbc/results/pipeline/reg.$set neuralreg /home/marcosbc/reg_model/model1.dy

done




