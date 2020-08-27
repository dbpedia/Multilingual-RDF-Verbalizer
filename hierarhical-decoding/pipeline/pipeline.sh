
moses="/home/marcosbc/LinearAMR/mosesdecoder"
lng="en"

pipeline_dir="/home/marcosbc/results/pipeline"
project_dir="/home/marcosbc/Multilingual-RDF-Verbalizer/pytorch"
model_dir="/home/marcosbc/results/output.03082020"

python $project_dir/Train.py -train-src $project_dir/data/en/ordering/train.src -train-tgt $project_dir/data/en/ordering/train.trg -dev-src $project_dir/data/en/ordering/dev.src -dev-tgt $project_dir/data/en/ordering/dev.trg -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 -gpu -eval $project_dir/data/en/end2end/dev.eval -test $project_dir/data/en/end2end/test.eval -save-dir $model_dir/ordering/ -tie-embeddings -translate -beam-size 5

for set in dev test
do
mv $model_dir/ordering/$set.eval.0.out $pipeline_dir/ordering.$set

python $project_dir/mapping.py $project_dir/data/en/end2end/$set.eval $pipeline_dir/ordering.$set ordering $pipeline_dir/ordering.mapped.$set
done

python $project_dir/Train.py -train-src $project_dir/data/en/structing/train.src -train-tgt $project_dir/data/en/structing/train.trg -dev-src $project_dir/data/en/structing/dev.src -dev-tgt $project_dir/data/en/structing/dev.trg -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 -gpu -eval $pipeline_dir/ordering.mapped.dev -test $pipeline_dir/ordering.mapped.test -save-dir $model_dir/structuring/ -tie-embeddings -translate -beam-size 5

for set in dev test
do
mv $model_dir/structuring/ordering.mapped.$set.0.out $pipeline_dir/structuring.$set

python $project_dir/mapping.py $pipeline_dir/ordering.mapped.$set $pipeline_dir/structuring.$set structing $pipeline_dir/structuring.mapped.$set
done

python $project_dir/Train.py -train-src $project_dir/data/en/lexicalization/train.src -train-tgt $project_dir/data/en/lexicalization/train.bpe.trg -dev-src $project_dir/data/en/lexicalization/dev.src -dev-tgt $project_dir/data/en/lexicalization/dev.bpe.trg -mtl -batch-size 32 -max-length 180 -lr 0.0005 -seed 13 -hidden-size 512 -enc-layers 4 -dec-layers 4 -enc-filter-size 2048 -dec-filter-size 2048 -enc-num-heads 8 -dec-num-heads 8 -enc-dropout 0.1 -dec-dropout 0.1 -steps 200000 -eval-steps 5000 -print-every 1000 -warmup-steps 8000 -gpu -eval $pipeline_dir/structuring.mapped.dev -test $pipeline_dir/structuring.mapped.test -save-dir $model_dir/lexicalization/ -tie-embeddings -translate -beam-size 5

for set in dev test
do
mv $model_dir/lexicalization/structuring.mapped.$set.0.out $pipeline_dir/lex.$set

python $project_dir/postProcessing.py -i $pipeline_dir/lex.$set -o $pipeline_dir/lexicalization.lower.$set

$moses/scripts/recaser/recase.perl --in $pipeline_dir/lexicalization.lower.$set --model $project_dir/data/en/lexicalization/case_model/moses.ini --moses $moses/bin/moses > $pipeline_dir/lexicalization.cs.$set

python $project_dir/utils/generate.py $pipeline_dir/lexicalization.cs.$set $pipeline_dir/ordering.mapped.$set $pipeline_dir/reg.$set neuralreg /home/marcosbc/reg_model/model1.dy

python $project_dir/utils/realization.py $pipeline_dir/reg.$set $pipeline_dir/realization.$set $project_dir/data/en/lexicalization/surfacevocab.json

$moses/scripts/tokenizer/normalize-punctuation.perl -l $lng < $pipeline_dir/realization.$set > $pipeline_dir/realization.punc.$set
$moses/scripts/tokenizer/detokenizer.perl -l $lng < $pipeline_dir/realization.punc.$set > $pipeline_dir/realization.detok.$set
$moses/scripts/recaser/detruecase.perl < $pipeline_dir/realization.detok.$set > $pipeline_dir/realization.post.$set

done
