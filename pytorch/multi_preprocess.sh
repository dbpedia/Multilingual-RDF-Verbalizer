
data="/home/msobrevillac/Projects/gsoc/Multilingual-RDF-Verbalizer/pytorch/data"

for split in train dev; do
	#awk '$0="<ordering> "$0' $data/ordering/$split.src > $data/ordering/multi.$split.src
	#awk '$0="<structuring> "$0' $data/structing/$split.src > $data/structing/multi.$split.src
	#awk '$0="<lexicalization> "$0' $data/lexicalization/$split.src > $data/lexicalization/multi.$split.src
	#awk '$0="<end2end> "$0' $data/end2end/$split.src > $data/end2end/multi.$split.src

	cat $data/ordering/multi.$split.src $data/structing/multi.$split.src $data/lexicalization/multi.$split.src $data/end2end/multi.$split.src > $data/multibpe/$split.src

#	cat $data/ordering/$split.trg $data/structing/$split.trg $data/lexicalization/$split.bpe.1.trg $data/end2end/$split.bpe.1.trg > $data/multi/$split.trg
	cat $data/ordering/$split.bpe.all.trg $data/structing/$split.bpe.all.trg $data/lexicalization/$split.bpe.all.trg $data/end2end/$split.bpe.all.trg > $data/multibpe/$split.trg
done


# Processing dev and test eval

#for split in dev test; do

#	awk '$0="<ordering> "$0' $data/ordering/$split.eval > $data/ordering/multi.$split.eval
#	awk '$0="<structuring> "$0' $data/structing/$split.eval > $data/structing/multi.$split.eval
#	awk '$0="<lexicalization> "$0' $data/lexicalization/$split.eval > $data/lexicalization/multi.$split.eval
#	awk '$0="<end2end> "$0' $data/end2end/$split.eval > $data/end2end/multi.$split.eval

#	cat $data/ordering/multi.$split.eval $data/structing/multi.$split.eval $data/lexicalization/multi.$split.eval $data/end2end/multi.$split.eval > $data/multi/$split.eval

#done




