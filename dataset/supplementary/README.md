# Manual Revision of Original Triples

## Documentation

<https://webnlg-challenge.loria.fr/docs/#triple-modification>

## Properties

Revisions made to original properties are documented in `*_lexicon.csv` files.

The files have the following structure:

| Original property | Modified property | Example                                                    |
| ----------------- |-------------------| -----------------------------------------------------------|
| open              | inaugurationDate  | Atatürk_Monument_(İzmir) \| open \| "1932-07-27"^^xsd:date |


Empty cell in the second column means that triples containing the original property were deleted and not used for further data creation.


## Entities (subjects and objects)

Revisions made to original subjects and objects are documented in `dbp-substitute-*.txt` files.

The files have the following structure: `original entity | modified entity`.

```
Airman_(comics) | Airman_(comicsCharacter)
```

## Removals

`dbp-remove-triples_*.txt` listed triples that were removed and not used for data collection.

The most common reason for deletion was that they had erroneous data. For example,

```
A_Glastonbury_Romance | isbn | 0
```
