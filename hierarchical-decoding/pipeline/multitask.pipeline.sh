moses="/home/marcosbc/LinearAMR/mosesdecoder"
lng="en"

project_dir="/home/marcosbc/Multilingual-RDF-Verbalizer/pytorch/"

multitask_model="/home/marcosbc/model/multitask"
pipeline_dir="/home/marcosbc/results/pipeline"
neuralreg_dir="/home/marcosbc/reg_model/model1.dy"


for set in dev test
do
python $project_dir/Translate.py -input $project_dir/data/en/end2end/$set.eval -gpu \
				-params $multitask_model/args.json -model $multitask_model/model.pt \
				-beam-size 5 -save-dir $pipeline_dir/ordering.$set -seed 13 \
				-src-vocab $project_dir/vocab/tied.vocab.json -mtl -task-id 0

python $project_dir/utils/mapping.py $project_dir/data/en/end2end/$set.eval $pipeline_dir/ordering.$set ordering $pipeline_dir/ordering.mapped.$set

python $project_dir/Translate.py -input $pipeline_dir/ordering.mapped.$set -gpu \
				-params $multitask_model/args.json -model $multitask_model/model.pt \
				-beam-size 5 -save-dir $pipeline_dir/structuring.$set -seed 13 \
				-src-vocab $project_dir/vocab/tied.vocab.json -mtl -task-id 1

python $project_dir/utils/mapping.py $pipeline_dir/ordering.mapped.$set $pipeline_dir/structuring.$set structing $pipeline_dir/structuring.mapped.$set

python $project_dir/Translate.py -input $pipeline_dir/structuring.mapped.$set -gpu \
				-params $multitask_model/args.json -model $multitask_model/model.pt \
				-beam-size 5 -save-dir $pipeline_dir/lex.$set -seed 13 \
				-src-vocab $project_dir/vocab/tied.vocab.json -mtl -task-id 2

python $project_dir/utils/postProcessing.py -i $pipeline_dir/lex.$set -o $pipeline_dir/lexicalization.lower.$set

$moses/scripts/recaser/recase.perl --in $pipeline_dir/lexicalization.lower.$set --model $project_dir/data/en/lexicalization/case_model/moses.ini --moses $moses/bin/moses > $pipeline_dir/lexicalization.cs.$set

python $project_dir/utils/generate.py $pipeline_dir/lexicalization.cs.$set $pipeline_dir/ordering.mapped.$set $pipeline_dir/reg.$set neuralreg $neuralreg_dir

python $project_dir/utils/realization.py $pipeline_dir/reg.$set $pipeline_dir/realization.$set $project_dir/data/en/lexicalization/surfacevocab.json

$moses/scripts/tokenizer/normalize-punctuation.perl -l $lng < $pipeline_dir/realization.$set > $pipeline_dir/realization.punc.$set
$moses/scripts/tokenizer/detokenizer.perl -l $lng < $pipeline_dir/realization.punc.$set > $pipeline_dir/realization.detok.$set
$moses/scripts/recaser/detruecase.perl < $pipeline_dir/realization.detok.$set > $pipeline_dir/realization.post.$set

done
