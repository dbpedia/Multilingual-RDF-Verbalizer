
#data_dir="/home/msobrevillac/Projects/gsoc/Multilingual-RDF-Verbalizer/pytorch/data/end2end"
data_dir=$1
model=$2
bpe_model=$3

moses_scripts="/home/msobrevillac/Projects/gsoc/mosesdecoder/scripts"
bpe_scripts="/home/msobrevillac/Projects/gsoc/subword-nmt"

mkdir $data_dir/$model

# train recaser
$moses_scripts/recaser/train-recaser.perl --dir $data_dir/$model --corpus $data_dir/train.trg

tr '[:upper:]' '[:lower:]' < $data_dir/train.trg > $data_dir/train.lower.trg
tr '[:upper:]' '[:lower:]' < $data_dir/dev.trg > $data_dir/dev.lower.trg



# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=20000

#minimum number of times we need to have seen a character sequence in the training text before we merge it into one unit
#this is applied to each training text independently, even with joint BPE
bpe_threshold=50

mkdir $data_dir/$bpe_model
# train BPE
$bpe_scripts/learn_joint_bpe_and_vocab.py -i $data_dir/train.lower.trg --write-vocabulary $data_dir/train.vocab.trg -s $bpe_operations -o $data_dir/$bpe_model/model.bpe

# apply BPE
for prefix in train dev
 do
  $bpe_scripts/apply_bpe.py -c $data_dir/$bpe_model/model.bpe --vocabulary $data_dir/train.vocab.trg --vocabulary-threshold $bpe_threshold < $data_dir/$prefix.lower.trg > $data_dir/$prefix.bpe.trg
 done

