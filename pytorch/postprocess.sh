

data_dir="/home/msobrevillac/Projects/gsoc/Multilingual-RDF-Verbalizer/pytorch/data/lexicalization"
output_dir="/home/msobrevillac/Projects/gsoc/Multilingual-RDF-Verbalizer/pytorch/output/multibpe/lexicalization"
moses="/home/msobrevillac/Projects/gsoc/mosesdecoder"
model="case_model"
lng=en


for prefix in dev test
 do
	python3.6 /home/msobrevillac/Projects/gsoc/Multilingual-RDF-Verbalizer/pytorch/postProcessing.py -i $output_dir/$prefix.eval.0.out -o $output_dir/$prefix.lower.out

	$moses/scripts/recaser/recase.perl --in $output_dir/$prefix.lower.out --model $data_dir/$model/moses.ini --moses $moses/bin/moses > $output_dir/$prefix.cs.out

	$moses/scripts/tokenizer/normalize-punctuation.perl -l $lng < $output_dir/$prefix.cs.out > $output_dir/$prefix.punc.out
	$moses/scripts/tokenizer/detokenizer.perl -l $lng < $output_dir/$prefix.punc.out > $output_dir/$prefix.out

 done
