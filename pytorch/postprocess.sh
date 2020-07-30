

data_dir="/home/msobrevillac/Projects/gsoc/Multilingual-RDF-Verbalizer/pytorch/data/end2end"
$moses="/home/msobrevillac/Projects/gsoc/mosesdecoder"
model="case_model"

$moses/scripts/recaser/recase.perl --in $data_dir/train.lower.trg --model $data_dir/$model/moses.ini --moses $moses/bin/moses


script_dir=`dirname $0`
# temporary variables
. $script_dir/tmp
# variables (toolkits; source and target language)
. $script_dir/vars

main_dir=$script_dir/../

if [ "$task" = "end2end" ] || [ "$task" = "end2end_augmented" ];
then
  sed -r 's/\@\@ //g' |
  $moses_scripts/recaser/detruecase.perl |
  $moses_scripts/tokenizer/normalize-punctuation.perl -l $lng |
  $moses_scripts/tokenizer/detokenizer.perl -l $lng
elif [ "$task" = "lexicalization" ];
then
  sed -r 's/\@\@ //g'
  #$moses_scripts/recaser/detruecase.perl |
  #$moses_scripts/tokenizer/detokenizer.perl -l $lng
else
  sed -r 's/\@\@ //g'
fi
