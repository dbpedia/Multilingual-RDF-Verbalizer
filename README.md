# neural-rdf-verbalizer-gsoc-2020
GSoC 2020 


##### Generating a Knowledge Subgraph for WebNLG dataset

```
python3.6 build_subgraph_from_webnlg.py \
   -webnlg <all train/dev/test split in the WebNLG dataset. Example: dataset/challenge2020_train_dev/en/train dataset/challenge2020_train_dev/en/dev > \
   -dbpedia <put all .ttl files of the Knowledge Base> \
   -max-depth <Max. depth path per instance> \
   -output <path for saving the Knowledge subgraph>
```
### Node embeddings generation
#### RDF2Vec

Folder rdf2vec contains the same code as [pyRDF2Vec] with some small modifications for generating node embeddings in the WebNLG.
To run:
```
    cd rdf2vec
    python3.6 execute.py 
       -dbpedia ../webnlg-dbpedia/subgraph_webnlg.ttl \
       -vocab vocabs/gat/eng/src_vocab -embed-size 300 -depth 8 \
       -algorithm rnd -walks 200 -jobs 5 -window 5 -sg \
       -max-iter 30 -negative 25 -min-count 1 -oe subgraph_embeddings -seed 13 \
       -webnlg dataset/challenge2020_train_dev/en/train -sup dataset/supplementary
```

#### PYKE

Folder pyke contains the same code as [PYKE] with some small modifications for generating node embeddings in the WebNLG.
To install: ./install_pyke.sh | conda activate pyke
To run:
```
    cd pyke
    python3.6 execute.py --kg_path KGs/webnlg-dbpedia/subgraph_webnlg.ttl \
        --embedding_dim 300 --num_iterations 1000 --K 45 --omega 0.45557 \
        --energy_release 0.0414 -webnlg ../Multilingual-RDF-Verbalizer/dataset/challenge2020_train_dev/en/train \
        -sup ../Multilingual-RDF-Verbalizer/dataset/supplementary
```


[pyRDF2Vec]: https://github.com/IBCNServices/pyRDF2Vec
[PYKE]: https://github.com/dice-group/PYKE
