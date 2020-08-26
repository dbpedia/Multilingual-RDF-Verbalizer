# WebNLG Challenge 2020 Data
WebNLG maps a set of RDF-triples to a text.
Each entry in WebNLG is a set of RDF-triples. Each set forms a tree.
Each entry has several references (a.k.a. lexicalisations).
Docs: https://webnlg-challenge.loria.fr/docs/

## WebNLG English
The challenge data includes version_2.1 (verbatim) and a new category "Company".

For versioning see here: https://gitlab.com/shimorina/webnlg-dataset

### Some statistics
|                     | Train  | Dev   |
| ------------------- |--------|-------|
| Entries             | 13,229 | 1,669 |  
| Lexicalisations     | 35,415 | 4,468 |
| Distinct properties | 385    | 300   |

## WebNLG Russian
Translation to Russian was based on the English WebNLG version 2.0 for 9 categories.

Additional features comparing to English:
- `<dbpedialinks>` were extracted from DBpedia between English and Russian entities by means of the property `sameAs`
- `<links>` were created manually for some entities to serve as pointers to translators
    There are two types of them:
    * `Spaniards | sameAs | испанцы`
    * `Tomatoes, guanciale, cheese, olive oil | includes | гуанчиале`
    (mostly used for string literals to translate some parts of them)

### Some statistics
|                     | Train  | Dev   |
| ------------------- |--------|-------|
| Entries             | 5,585  | 793   |  
| Lexicalisations     | 14,612 | 2,068 |
| Distinct properties | 229    | 119   |

Note: Different English texts can be translated to the same text in Russian.
Since original English sentences are kept in the data, the number of **distinct** lexicalisations in Russian is a bit lower than the total number reported above.

### Note on Russian Data
#### Short version
Russian WebNLG was translated from English with an MT system, and then post-edited using crowdsourcing on the Yandex.Toloka platform.
Afterwards, some sanity checks and spellchecker were ran to ensure data quality.
However, some texts may still be lacking some fluency, etc.

#### Long version
The Russian data creation followed the procedure below:

1. Russian WebNLG was translated from English WebNLG version 2.0 with an MT system (see [paper](https://www.aclweb.org/anthology/W19-3706.pdf))

2. Then it was post-edited using crowdsourcing on the Yandex.Toloka platform in two steps:

    * we asked people to post-edit and provided them with some pointers for translation of entities (those translations were made by experts). In the Russian dataset, there are <links> and <dbpedialinks>: they are the pointers that were used. Crowdworkers were asked to use the pointers as much as possible.
    * verification step: given post-edited sentences, we asked people to check if everything is fine with translation (grammar, spellchecking, etc) and if entity translation was correct. If the translation was detected as erroneous, it was moved to the first step again.

3. Afterwards, some sanity checks and spellchecker were ran to ensure data quality. All the detected cases were then manually verified by experts (Russian native speakers), and they edited texts one more time if needed.

So we suppose that the Russian data is of a decent quality, however, some texts may still be lacking some fluency. Those mainly stem from the original English texts that were not fluent.

#### Russian vs. English data
Note that English challenge data is based on WebNLG v2.1, while Russian data was translated from English WebNLG v2.0. So English texts in Russian and English challenge data differ.

## Other resources
 * Enriched WebNLG-en (for webnlg2017 version; covers 9 categories): https://github.com/ThiagoCF05/webnlg

## XML data reader
 * WebNLG corpus reader: https://gitlab.com/webnlg/corpus-reader

## Test data
Test sets will be released for:
 * the categories seen in the training data (Astronaut, City, University, etc);
 * several new unseen categories (categories not included in the training data, i.e., suprise domains).

## Feedback
If you spot an error somewhere, don't hesitate to write to us: <webnlg-challenge@inria.fr>. We will correct errors for the next corpus release. Let's make NLP datasets better together!

## License
The WebNLG data is licensed under the following license: CC Attribution-Noncommercial-Share Alike 4.0 International.

