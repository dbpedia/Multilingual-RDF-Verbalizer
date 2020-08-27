

#data_dir="/home/msobrevillac/Projects/gsoc/Multilingual-RDF-Verbalizer/pytorch/data/ru/end2end"
#output_dir="/home/msobrevillac/Projects/gsoc/Multilingual-RDF-Verbalizer/pytorch/output/ru.end2end"
#model="caser_model"
#lng=ru
input=$1
recaser_model=$2
lng=$3

moses="/home/msobrevillac/Projects/gsoc/mosesdecoder"

python3.6 postProcessing.py -i $input -o $input.lower.out

$moses/scripts/recaser/recase.perl --in $input.lower.out --model $recaser_model/moses.ini --moses $moses/bin/moses > $input.cs.out

$moses/scripts/tokenizer/normalize-punctuation.perl -l $lng < $input.cs.out > $input.punc.out
$moses/scripts/tokenizer/detokenizer.perl -l $lng < $input.punc.out > $input.out

