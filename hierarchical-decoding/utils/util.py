import random
import numpy as np
import torch
import torch.nn as nn
import re
import json
import sys

def initialize_weights(m):
    '''
        Initialize the weights of a model
        m: model
    '''
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def set_seed(seed):
    '''
        Setting a seed to make our experiments reproducible
        seed: seed value
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    '''
        Count the number of parameters that requires grad
        model: model
    '''

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def split_triples(text):
    triples, triple = [], []
    for w in text:
        if w not in ['<TRIPLE>', '</TRIPLE>']:
            triple.append(w)
        elif w == '</TRIPLE>':
            triples.append(triple)
            triple = []
    return triples

def join_triples(triples):
    result = []
    for triple in triples:
        result.append('<TRIPLE>')
        result.extend(triple)
        result.append('</TRIPLE>')
    return result

def delexicalize(triples):
    entities = {}
    entity_pos = 1
    for triple in triples:
        agent = triple[0]
        if agent not in entities:
            entities[agent] = 'ENTITY-' + str(entity_pos)
            entity_pos += 1
        triple[0] = entities[agent]

        patient = triple[-1]
        if patient not in entities:
            entities[patient] = 'ENTITY-' + str(entity_pos)
            entity_pos += 1
        triple[-1] = entities[patient]

    return triples

def entity_mapping(triples):
    entitytag = {}
    entities = {}
    entity_pos = 1
    for triple in triples:
        agent = triple[0]
        if agent not in entitytag:
            entitytag[agent] = 'ENTITY-' + str(entity_pos)
            entities['ENTITY-' + str(entity_pos)] = agent
            entity_pos += 1

        patient = triple[-1]
        if patient not in entitytag:
            entitytag[patient] = 'ENTITY-' + str(entity_pos)
            entities['ENTITY-' + str(entity_pos)] = patient
            entity_pos += 1

    return entities

def split_struct(text):
    sentences, triples, triple = [], [], []
    for w in text:
        if w not in ['<SNT>', '</SNT>', '<TRIPLE>', '</TRIPLE>']:
            triple.append(w)
        elif w == '</TRIPLE>':
            triples.append(triple)
            triple = []
        elif w == '</SNT>':
            sentences.append(triples)
            triples = []
    return sentences

def join_struct(sentences):
    result = []
    for sentence in sentences:
        result.append('<SNT>')
        for triple in sentence:
            result.append('<TRIPLE>')
            result.extend(triple)
            result.append('</TRIPLE>')
        result.append('</SNT>')
    return result

def delexicalize_struct(struct):
    entities, entity_pos = {}, 1
    for sentence in struct:
        for triple in sentence:
            agent = triple[0]
            if agent not in entities:
                entities[agent] = 'ENTITY-' + str(entity_pos)
                entity_pos += 1
            triple[0] = entities[agent]

            patient = triple[-1]
            if patient not in entities:
                entities[patient] = 'ENTITY-' + str(entity_pos)
                entity_pos += 1
            triple[-1] = entities[patient]

    return struct

def delexicalize_verb(template):
    regex = r'(tense=|person=)(.*?),'
    template = re.sub(regex, r'\1null,', template)

    regex = r'(number=)(.*?)]'
    return re.sub(regex, r'\1null]', template)


def load_params(params_file):
    return json.load(open(params_file))

def save_params(args, params_file):
    params = {}
    params['hidden_size'] = args.hidden_size
    params['encoder_ff_size'] = args.encoder_ff_size
    params['encoder_layer'] = args.encoder_layer
    params['encoder_head'] = args.encoder_head
    params['encoder_dropout'] = args.encoder_dropout
    params['decoder_ff_size'] = args.decoder_ff_size
    params['decoder_layer'] = args.decoder_layer
    params['decoder_head'] = args.decoder_head
    params['decoder_dropout'] = args.decoder_dropout
    params['max_length'] = args.max_length
    params['tie_embeddings'] = args.tie_embeddings

    if args.mtl:
        params['number_encoder'] = 1
        params['number_decoder'] = len(args.train_target)
    else:
        params['number_encoder'] = len(args.train_source)
        params['number_decoder'] = len(args.train_target)

    with open(params_file, "w") as outfile: 
        json.dump(params, outfile)


def build_vocab(files, vocabulary=None, mtl=False, name="src", save_dir="/"):
    '''
        This method builds the vocabulary
        files: files to generate the vocabulary.
        vocabulary: if there is a vocabulary we should only load it
        mtl: if true we should generate an specific vocabulary for each file. Otherwise, a joint vocabulary
        name: prefix of the vocabulary
        save_dir: folder where the vocabular will be saved
    '''

    vocabs = []
    from .vocab import Vocab
    if vocabulary is not None:
        for v in vocabulary:
            print(f'Loading from {v}')
            vb = Vocab()
            vb.load_from_file(v)
            vocabs.append(vb)
    else:
        if mtl is True:
            for index, f in enumerate(files):
                vb = Vocab()
                vb.build_vocab([f])
                vb.save(save_dir + name + ".vocab." + str(index) + ".json")
                vocabs.append(vb)
        else:
            vb = Vocab()
            vb.build_vocab(files)
            vb.save(save_dir + name + ".vocab.json")
            vocabs.append(vb)

    for index, vb in enumerate(vocabs):
        print(f'vocabulary size {index+1:d}: {vb.len():d}')

    return vocabs
    



