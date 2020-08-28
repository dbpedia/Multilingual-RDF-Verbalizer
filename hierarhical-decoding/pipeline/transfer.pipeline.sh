moses="/home/marcosbc/LinearAMR/mosesdecoder"
lng="en"

pipeline_dir="/home/marcosbc/results/transfer.pipeline"
project_dir="/home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/"
model_dir="/home/marcosbc/results/output.11082020"
neuralreg_dir="/home/marcosbc/reg_model/model1.dy"

python Train.py -train-src data/en/ordering/train.src -train-tgt data/en/ordering/train.trg -dev-src data/en/ordering/dev.src -dev-tgt data/en/ordering/dev.trg -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 -gpu -eval data/en/end2end/dev.eval -test data/en/end2end/test.eval -save-dir $model_dir/tr.ordering/ -tie-embeddings -translate -beam-size 5 -src-vocab vocab/tied.vocab.json

for set in dev test
do
mv $model_dir/tr.ordering/$set.eval.0.out $pipeline_dir/ordering.$set

python mapping.py data/en/end2end/$set.eval $pipeline_dir/ordering.$set ordering $pipeline_dir/ordering.mapped.$set
done

python Train.py -train-src data/en/structing/train.src -train-tgt data/en/structing/train.trg -dev-src data/en/structing/dev.src -dev-tgt data/en/structing/dev.trg -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 -gpu -eval $pipeline_dir/ordering.mapped.dev -test $pipeline_dir/ordering.mapped.test -save-dir $model_dir/tr.structuring/ -tie-embeddings -translate -beam-size 5 -src-vocab vocab/tied.vocab.json

for set in dev test
do
mv $model_dir/tr.structuring/ordering.mapped.$set.0.out $pipeline_dir/structuring.$set

python mapping.py $pipeline_dir/ordering.mapped.$set $pipeline_dir/structuring.$set structing $pipeline_dir/structuring.mapped.$set
done

python Train.py -train-src data/en/lexicalization/train.src -train-tgt data/en/lexicalization/train.bpe.1.trg -dev-src data/en/lexicalization/dev.src -dev-tgt data/en/lexicalization/dev.bpe.1.trg -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 -gpu -eval $pipeline_dir/structuring.mapped.dev -test $pipeline_dir/structuring.mapped.test -save-dir $model_dir/tr.lexicalization/ -tie-embeddings -translate -beam-size 5 -src-vocab vocab/tied.vocab.json

for set in dev test
do
mv $model_dir/tr.lexicalization/structuring.mapped.$set.0.out $pipeline_dir/lex.$set

python postProcessing.py -i $pipeline_dir/lex.$set -o $pipeline_dir/lexicalization.lower.$set

$moses/scripts/recaser/recase.perl --in $pipeline_dir/lexicalization.lower.$set --model data/en/lexicalization/case_model/moses.ini --moses $moses/bin/moses > $pipeline_dir/lexicalization.cs.$set

python utils/generate.py $pipeline_dir/lexicalization.cs.$set $pipeline_dir/ordering.mapped.$set $pipeline_dir/reg.$set neuralreg $neuralreg_dir

python utils/realization.py $pipeline_dir/reg.$set $pipeline_dir/realization.$set data/en/lexicalization/surfacevocab.json

$moses/scripts/tokenizer/normalize-punctuation.perl -l $lng < $pipeline_dir/realization.$set > $pipeline_dir/realization.punc.$set
$moses/scripts/tokenizer/detokenizer.perl -l $lng < $pipeline_dir/realization.punc.$set > $pipeline_dir/realization.detok.$set
$moses/scripts/recaser/detruecase.perl < $pipeline_dir/realization.detok.$set > $pipeline_dir/realization.post.$set

done




