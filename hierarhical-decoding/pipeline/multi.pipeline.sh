
moses="/home/marcosbc/LinearAMR/mosesdecoder"
lng="en"

pipeline_dir="/home/marcosbc/results/multi-pipeline"
project_dir="/home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/"
model_dir="/home/marcosbc/results/output.11082020/multi"
neuralreg_dir="/home/marcosbc/reg_model/model1.dy"

awk '$0="<ordering> "$0' data/en/end2end/dev.eval > data/en/end2end/dev.multi.eval
awk '$0="<ordering> "$0' data/en/end2end/test.eval > data/en/end2end/test.multi.eval

python Train.py -train-src data/en/multi/train.src -train-tgt data/en/multi/train.trg -dev-src data/en/multi/dev.src -dev-tgt data/en/multi/dev.trg -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 -gpu -eval data/en/end2end/dev.multi.eval -test data/en/end2end/test.multi.eval -save-dir $model_dir/ -tie-embeddings -translate -beam-size 5

for set in dev test
do
mv $model_dir/$set.multi.eval.0.out $pipeline_dir/ordering.$set

python mapping.py data/en/end2end/$set.eval $pipeline_dir/ordering.$set ordering $pipeline_dir/ordering.mapped.$set

awk '$0="<structuring> "$0' $pipeline_dir/ordering.mapped.$set > $pipeline_dir/ordering.mapped.multi.$set
done


python Train.py -train-src data/en/multi/train.src -train-tgt data/en/multi/train.trg -dev-src data/en/multi/dev.src -dev-tgt data/en/multi/dev.trg -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 -gpu -eval $pipeline_dir/ordering.mapped.multi.dev -test $pipeline_dir/ordering.mapped.multi.test -save-dir $model_dir/ -tie-embeddings -translate -beam-size 5

for set in dev test
do
mv $model_dir/ordering.mapped.multi.$set.0.out $pipeline_dir/structuring.$set

python mapping.py $pipeline_dir/ordering.mapped.$set $pipeline_dir/structuring.$set structing $pipeline_dir/structuring.mapped.$set

awk '$0="<lexicalization> "$0' $pipeline_dir/structuring.mapped.$set > $pipeline_dir/structuring.mapped.multi.$set
done

python Train.py -train-src data/en/multi/train.src -train-tgt data/en/multi/train.trg -dev-src data/en/multi/dev.src -dev-tgt data/en/multi/dev.trg -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 -gpu -eval $pipeline_dir/structuring.mapped.multi.dev -test $pipeline_dir/structuring.mapped.multi.test -save-dir $model_dir/ -tie-embeddings -translate -beam-size 5

for set in dev test
do
mv $model_dir/structuring.mapped.multi.$set.0.out $pipeline_dir/lex.$set

python postProcessing.py -i $pipeline_dir/lex.$set -o $pipeline_dir/lexicalization.lower.$set

$moses/scripts/recaser/recase.perl --in $pipeline_dir/lexicalization.lower.$set --model data/en/lexicalization/case_model/moses.ini --moses $moses/bin/moses > $pipeline_dir/lexicalization.cs.$set

python utils/generate.py $pipeline_dir/lexicalization.cs.$set $pipeline_dir/ordering.mapped.$set $pipeline_dir/reg.$set neuralreg $neuralreg_dir

python utils/realization.py $pipeline_dir/reg.$set $pipeline_dir/realization.$set data/en/lexicalization/surfacevocab.json

$moses/scripts/tokenizer/normalize-punctuation.perl -l $lng < $pipeline_dir/realization.$set > $pipeline_dir/realization.punc.$set
$moses/scripts/tokenizer/detokenizer.perl -l $lng < $pipeline_dir/realization.punc.$set > $pipeline_dir/realization.detok.$set
$moses/scripts/recaser/detruecase.perl < $pipeline_dir/realization.detok.$set > $pipeline_dir/realization.post.$set

done
