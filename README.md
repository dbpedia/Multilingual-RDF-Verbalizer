# Multilingual RDF verbalizer - GSoC/2020

This repository contains the code of the Google Summer of Code project.

Author - [Marco Sobrevilla]

Abstract :

This project aims to create a natural language generation framework that verbalizes RDF triples.

An RDF triple set contains a triple set, each of the form < subject | predicate | object>, the model aims to take in a set of such triples and output the information in human-readable form.

For ex : < Marco_Sobrevilla | birthplace | Lima > < Marco Sobrevilla | lives in | Brazil > output: Marco Sobrevilla was born in Lima, and lives in Brazil. The model must be capable of doing the same in multiple languages, hence the name multilingual RDF verbalizer.

In particular, this work is divided in two parts:

- The [first one] consists in exploring the use of pre-trained node embeddings into the previous GSoC project which uses Graph Attention Network to encode the triple set and a Transformer to decode its respective surface realisation. The s
- The [second one] consists in exploring different ways to approach hierarchical decoding, i.e., execute each task (discourse ordering, text structuring and lexicalisation tasks) sequentially.

You can see my GSoC posts in this [link].

[Marco Sobrevilla]: https://github.com/msobrevillac
[first one]: https://github.com/dbpedia/Multilingual-RDF-Verbalizer/tree/master/node-embeddings
[second one]: https://github.com/dbpedia/Multilingual-RDF-Verbalizer/tree/master/hierarchical-decoding
[link]: https://medium.com/@msobrevillac/week-12-google-summer-of-code-2020-thats-all-folks-a1f70aa16589?sk=aabc83bb760c6be9703b7c550cfe6167

