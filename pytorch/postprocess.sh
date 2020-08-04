

data_dir="/home/msobrevillac/Projects/gsoc/Multilingual-RDF-Verbalizer/pytorch/data/lexicalization"
output_dir="/home/msobrevillac/Projects/gsoc/Multilingual-RDF-Verbalizer/pytorch/output/lexicalization"
moses="/home/msobrevillac/Projects/gsoc/mosesdecoder"
model="case_model"
lng=en

python3.6 /home/msobrevillac/Projects/gsoc/Multilingual-RDF-Verbalizer/pytorch/postProcessing.py

$moses/scripts/recaser/recase.perl --in $output_dir/dev.lower.out --model $data_dir/$model/moses.ini --moses $moses/bin/moses > $output_dir/dev.cs.out
$moses/scripts/recaser/recase.perl --in $output_dir/test.lower.out --model $data_dir/$model/moses.ini --moses $moses/bin/moses > $output_dir/test.cs.out


$moses/scripts/tokenizer/normalize-punctuation.perl -l $lng < $output_dir/dev.cs.out > $output_dir/dev.punc.out
$moses/scripts/tokenizer/detokenizer.perl -l $lng < $output_dir/dev.punc.out > $output_dir/dev.out


$moses/scripts/tokenizer/normalize-punctuation.perl -l $lng < $output_dir/test.cs.out > $output_dir/test.punc.out
$moses/scripts/tokenizer/detokenizer.perl -l $lng < $output_dir/test.punc.out > $output_dir/test.out
